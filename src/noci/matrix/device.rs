// noci/matrix/device.rs
//! `DeviceNOCIRows`, the GPU counterpart of `contract_noci_rows` in `matrix.rs` for the generalised
//! Fock and the fused Hamiltonian and overlap, with its kernels in `device.cu`. It contracts
//! `y_l = \sum_r O_{lr} c_r` over the determinant pairs of the reference pairs `DeviceWicks` holds
//! and the same-parent pairs of every orthonormal parent `DeviceOrthogonal` holds. `fock` and
//! `hamiltonian` tell `contract_noci_rows` which reference pairs to leave out for each operator, so
//! the host and GPU rows add up to the whole contraction. Real scalars only.

// External crate imports.
use cudarc::driver::{CudaFunction, DriverError, LaunchConfig, PushKernelArg};
use ndarray::Array1;
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::{DeviceDeterminant, NOCIIndex};
use crate::elements::nonorthogonalwicks::DeviceWicks;
use crate::elements::{DeviceOrthogonal, FockData, NOCIData};
use crate::gpu::Device;

/// Kernels of `device.cu`, compiled by `build/build.rs`.
const KERNELS: &str = include_str!(concat!(env!("OUT_DIR"), "/noci/matrix/device.ptx"));

/// Threads per block, matching `THREADS` in `device.cu`.
const THREADS: u32 = 128;

/// Row contractions of the GPU-evaluated determinant pairs.
pub(crate) struct DeviceNOCIRows {
    /// GPU and the stream every transfer and launch is issued on.
    device: Device,
    /// Wick intermediates of the GPU-evaluated pairs of distinct references.
    wicks: DeviceWicks,
    /// Slater-Condon integrals of the orthonormal parents.
    orthogonal: DeviceOrthogonal,
    /// Whether the GPU contracts the Fock elements of each ordered reference pair `x * nref + w`.
    pub(crate) fock: Vec<bool>,
    /// Whether the GPU contracts the Hamiltonian and overlap elements of each ordered reference
    /// pair `x * nref + w`.
    pub(crate) hamiltonian: Vec<bool>,
    /// Kernel `contract_fock_rows`.
    contract_fock: CudaFunction,
    /// Kernel `contract_hamiltonian_overlap_rows`.
    contract_hamiltonian: CudaFunction,
}

impl DeviceNOCIRows {
    /// Copy the Wick intermediates and orthogonal integrals of the current generalised Fock to the
    /// first GPU. Returns `None` without Wick intermediates or for complex scalars.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates.
    /// - `fock`: Current generalised-Fock data.
    /// # Returns:
    /// - `Option<Result<DeviceNOCIRows, DriverError>>`: The GPU rows, a driver error, or `None`
    ///   when the contraction stays on the host.
    pub(crate) fn new<T: NOCIScalar>(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Option<Result<Self, DriverError>> {
        // A real scalar keeps only the real part of `i`.
        let real = T::from_complex(Complex64::i()).im() == 0.0;
        if !(real && data.input.wicks.enabled && data.wicks.is_some()) {
            return None;
        }

        Some(Self::upload(data, fock))
    }

    /// Open the first GPU, copy the intermediates and integrals and load the kernels.
    /// # Arguments:
    /// - `data`: Shared NOCI data with Wick intermediates.
    /// - `fock`: Current generalised-Fock data.
    /// # Returns:
    /// - `Result<DeviceNOCIRows, DriverError>`: The GPU rows.
    fn upload<T: NOCIScalar>(
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Result<Self, DriverError> {
        let device = Device::new(0)?;
        let view = data.wicks.expect("GPU rows need Wick intermediates");
        let wicks = DeviceWicks::new(&device.stream, view)?;
        let orthogonal = DeviceOrthogonal::new(&device.stream, data, fock)?;
        let module = device.load(KERNELS)?;

        // Distinct references through Wick's theorem, one orthonormal parent by Slater-Condon rules.
        let nref = wicks.nref;
        let evaluated = |same: &[bool]| {
            (0..nref * nref)
                .map(|k| wicks.parents[k] || (k / nref == k % nref && same[k / nref]))
                .collect::<Vec<bool>>()
        };

        Ok(Self {
            fock: evaluated(&orthogonal.fock),
            hamiltonian: evaluated(&orthogonal.hamiltonian),
            contract_fock: module.load_function("contract_fock_rows")?,
            contract_hamiltonian: module.load_function("contract_hamiltonian_overlap_rows")?,
            device,
            wicks,
            orthogonal,
        })
    }

    /// Contract generalised-Fock rows, `y_l = \sum_r (F_{lr} - shift S_{lr}) c_r`, over the
    /// GPU-evaluated reference pairs, the counterpart of `contract_noci_rows` with `Fock`.
    /// # Arguments:
    /// - `self`: GPU rows.
    /// - `data`: Shared NOCI data holding the determinant space.
    /// - `left`: Row determinants.
    /// - `right`: Column determinants, the order of `c`.
    /// - `c`: Coefficients of the column determinants.
    /// - `shift`: Overlap shift of the Fock operator.
    /// # Returns:
    /// - `Result<Vec<f64>, DriverError>`: Contracted rows in the order of `left`.
    pub(crate) fn contract_fock_rows<T: NOCIScalar>(
        &self,
        data: &NOCIData<'_, T>,
        left: &[NOCIIndex],
        right: &[NOCIIndex],
        c: &Array1<T>,
        shift: f64,
    ) -> Result<Vec<f64>, DriverError> {
        if left.is_empty() {
            return Ok(Vec::new());
        }

        let stream = &self.device.stream;
        let bras = left
            .iter()
            .map(|&i| DeviceDeterminant::new(data.space, i))
            .collect::<Vec<_>>();
        let kets = right
            .iter()
            .map(|&i| DeviceDeterminant::new(data.space, i))
            .collect::<Vec<_>>();
        let coefficients = c.iter().map(|v| v.re()).collect::<Vec<f64>>();

        let bras = stream.clone_htod(&bras)?;
        let kets = stream.clone_htod(&kets)?;
        let coefficients = stream.clone_htod(&coefficients)?;
        let mut y = stream.alloc_zeros::<f64>(left.len())?;
        let (nref, nright) = (self.wicks.nref as u64, right.len() as u64);

        let mut launch = stream.launch_builder(&self.contract_fock);
        launch.arg(&self.wicks.pairs).arg(&self.wicks.combinations);
        launch.arg(&nref).arg(&self.wicks.slab);
        launch
            .arg(&self.orthogonal.parents)
            .arg(&self.orthogonal.integrals);
        launch.arg(&bras).arg(&kets).arg(&coefficients).arg(&nright);
        launch.arg(&shift).arg(&mut y);
        // One block of `THREADS` per bra row.
        let rows = LaunchConfig {
            grid_dim: (left.len() as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(rows) }?;

        stream.clone_dtoh(&y)
    }

    /// Contract Hamiltonian and overlap rows, `v_l = \sum_r H_{lr} c_r` and
    /// `s_l = \sum_r S_{lr} c_r`, over the GPU-evaluated reference pairs, the counterpart of
    /// `contract_noci_rows` with `HamiltonianOverlap`.
    /// # Arguments:
    /// - `self`: GPU rows.
    /// - `data`: Shared NOCI data holding the determinant space and the nuclear repulsion.
    /// - `left`: Row determinants.
    /// - `right`: Column determinants, the order of `c`.
    /// - `c`: Coefficients of the column determinants.
    /// # Returns:
    /// - `Result<(Vec<f64>, Vec<f64>), DriverError>`: Contracted Hamiltonian and overlap rows in
    ///   the order of `left`.
    pub(crate) fn contract_hamiltonian_overlap_rows<T: NOCIScalar>(
        &self,
        data: &NOCIData<'_, T>,
        left: &[NOCIIndex],
        right: &[NOCIIndex],
        c: &Array1<T>,
    ) -> Result<(Vec<f64>, Vec<f64>), DriverError> {
        if left.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let stream = &self.device.stream;
        let bras = left
            .iter()
            .map(|&i| DeviceDeterminant::new(data.space, i))
            .collect::<Vec<_>>();
        let kets = right
            .iter()
            .map(|&i| DeviceDeterminant::new(data.space, i))
            .collect::<Vec<_>>();
        let coefficients = c.iter().map(|v| v.re()).collect::<Vec<f64>>();

        let bras = stream.clone_htod(&bras)?;
        let kets = stream.clone_htod(&kets)?;
        let coefficients = stream.clone_htod(&coefficients)?;
        let mut v = stream.alloc_zeros::<f64>(left.len())?;
        let mut s = stream.alloc_zeros::<f64>(left.len())?;
        let (nref, nright) = (self.wicks.nref as u64, right.len() as u64);
        let enuc = data.ao.enuc;

        let mut launch = stream.launch_builder(&self.contract_hamiltonian);
        launch.arg(&self.wicks.pairs).arg(&self.wicks.combinations);
        launch.arg(&nref).arg(&self.wicks.slab);
        launch
            .arg(&self.orthogonal.parents)
            .arg(&self.orthogonal.integrals);
        launch.arg(&bras).arg(&kets).arg(&coefficients).arg(&nright);
        launch.arg(&enuc).arg(&mut v).arg(&mut s);
        // One block of `THREADS` per bra row.
        let rows = LaunchConfig {
            grid_dim: (left.len() as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(rows) }?;

        Ok((stream.clone_dtoh(&v)?, stream.clone_dtoh(&s)?))
    }
}
