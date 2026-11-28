// noci/factorise/onebody/device.rs
//! `DeviceOneBody`, the GPU counterpart of `OneBodyFactorisation` in `onebody.rs`, with its kernels
//! in `device.cu`. Each type and method mirrors the host one of the same name: `DeviceOneBodyBlock`
//! for `OneBodyBlock`, `DeviceOneBodyScratch` for `OneBodyScratch`, and `apply_one_body`,
//! `apply_one_body_factorised` and `apply_one_body_orthogonal` for the host methods. The factor
//! tables and the index arrays of the source and target determinants are uploaded once per
//! factorisation, and `apply_one_body` takes and returns vectors resident on the same GPU. Real
//! operators and vectors only; a factorisation with contraction-form or transient pairs, or with
//! complex factors, stays on the host.

// External crate imports.
use cudarc::driver::{CudaFunction, CudaSlice, DriverError, LaunchConfig, PushKernelArg};
use num_complex::Complex64;

// Crate-root imports.
use crate::NOCIScalar;
use crate::determinant::NOCIIndex;
use crate::elements::{
    FockData, NOCIData, calculate_f_pair_orthogonal, calculate_s_pair_orthogonal,
};
use crate::gpu::Device;

// Parent/sibling imports.
use super::{OneBodyBlock, OneBodyContraction, OneBodyFactorisation};

/// Kernels of `device.cu`, compiled by `build/build.rs`.
const KERNELS: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/noci/factorise/onebody/device.ptx"
));

/// Threads per warp, matching `WARP` in `device.cu`.
const WARP: u64 = 32;

/// Device counterpart of `FactorisedOneBodyBlock`: one factorised parent pair, with the first spin
/// the one contracted first: alpha for alpha-first pairs and beta for beta-first pairs.
struct DeviceFactorisedOneBodyBlock {
    /// First-spin Fock factors, `n_t1 x n_s1`.
    f1: CudaSlice<f64>,
    /// First-spin overlap factors, `n_t1 x n_s1`.
    s1: CudaSlice<f64>,
    /// Second-spin Fock factors, `n_t2 x n_s2`.
    f2: CudaSlice<f64>,
    /// Second-spin overlap factors, `n_t2 x n_s2`.
    s2: CudaSlice<f64>,
    /// Number of source first-spin strings.
    n_s1: u64,
    /// Number of target first-spin strings.
    n_t1: u64,
    /// Number of source second-spin strings.
    n_s2: u64,
    /// Start of each source second-spin string's determinants, `n_s2 + 1` entries.
    offsets: CudaSlice<u64>,
    /// First-spin string of each source determinant, grouped by second-spin string.
    first: CudaSlice<u64>,
    /// Vector position of each source determinant, grouped by second-spin string.
    sources: CudaSlice<u64>,
    /// Vector position of each target determinant.
    targets: CudaSlice<u64>,
    /// First-spin string of each target determinant.
    t1: CudaSlice<u64>,
    /// Second-spin string of each target determinant.
    t2: CudaSlice<u64>,
    /// Number of target determinants.
    ntarget: u64,
}

/// Device counterpart of `OrthogonalOneBodyBlock`: one same-parent orthogonal pair, as compressed
/// rows over its target determinants.
struct DeviceOrthogonalOneBodyBlock {
    /// Start of each row's single couplings, `ntarget + 1` entries.
    single_offsets: CudaSlice<u64>,
    /// Source position of each single coupling.
    single_source: CudaSlice<u64>,
    /// Fock element of each single coupling.
    single_f: CudaSlice<f64>,
    /// Start of each row's identical-occupation terms, `ntarget + 1` entries.
    same_offsets: CudaSlice<u64>,
    /// Source position of each identical-occupation term.
    same_source: CudaSlice<u64>,
    /// Fock element of each identical-occupation term.
    same_f: CudaSlice<f64>,
    /// Overlap element of each identical-occupation term.
    same_s: CudaSlice<f64>,
    /// Vector position of each target determinant.
    targets: CudaSlice<u64>,
    /// Number of target determinants.
    ntarget: u64,
}

/// Device counterpart of `OneBodyBlock`: one parent pair.
enum DeviceOneBodyBlock {
    /// Spin-factorised nonorthogonal pair.
    Factorised(Box<DeviceFactorisedOneBodyBlock>),
    /// Same-parent orthogonal pair.
    Orthogonal(Box<DeviceOrthogonalOneBodyBlock>),
}

/// First-contraction buffers, the counterpart of `OneBodyScratch`.
pub(crate) struct DeviceOneBodyScratch {
    /// First contraction `T^F` of the current factorised pair, sized for the largest pair.
    first_f: CudaSlice<f64>,
    /// First contraction `T^S` of the current factorised pair, sized for the largest pair.
    first_s: CudaSlice<f64>,
}

/// The spin-factorised one-body operator on a GPU.
pub(crate) struct DeviceOneBody {
    /// GPU and the stream every transfer and launch is issued on.
    pub(crate) device: Device,
    /// Parent pairs in the host factorisation's order.
    blocks: Vec<DeviceOneBodyBlock>,
    /// Length of the determinant vectors.
    n: usize,
    /// Largest first contraction `n_t1 x n_s2` of any factorised pair.
    nscratch: usize,
    /// Kernel `apply_one_body_factorised_first`.
    first: CudaFunction,
    /// Kernel `apply_one_body_factorised_second`.
    second: CudaFunction,
    /// Kernel `apply_one_body_orthogonal`.
    orthogonal: CudaFunction,
}

impl DeviceOneBody {
    /// Upload a one-body factorisation to the first GPU. Returns `None` when the factorisation
    /// cannot run there: complex scalars, or any pair in contraction form or with transient tables.
    /// # Arguments:
    /// - `one_body`: Host factorisation, with its factor tables built.
    /// - `data`: Shared NOCI data over the determinant space the factorisation was built on.
    /// - `fock`: Fock data the factorisation was built from.
    /// # Returns:
    /// - `Option<Result<DeviceOneBody, DriverError>>`: The device operator, a driver error, or
    ///   `None` when the factorisation stays on the host.
    pub(super) fn new<T: NOCIScalar>(
        one_body: &OneBodyFactorisation<T>,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Option<Result<Self, DriverError>> {
        let supported = one_body.blocks.iter().all(|block| {
            matches!(
                block,
                OneBodyBlock::Factorised(_) | OneBodyBlock::Orthogonal(_)
            )
        });
        // A real scalar keeps only the real part of `i`.
        let real = T::from_complex(Complex64::i()).im() == 0.0;
        if !real || !supported {
            return None;
        }

        Some(Self::upload(one_body, data, fock))
    }

    /// Upload every parent pair and load the kernels.
    /// # Arguments:
    /// - `one_body`: Host factorisation of real pairs in factorised or orthogonal form.
    /// - `data`: Shared NOCI data over the determinant space the factorisation was built on.
    /// - `fock`: Fock data the factorisation was built from.
    /// # Returns:
    /// - `Result<DeviceOneBody, DriverError>`: The device operator.
    fn upload<T: NOCIScalar>(
        one_body: &OneBodyFactorisation<T>,
        data: &NOCIData<'_, T>,
        fock: &FockData<'_, T>,
    ) -> Result<Self, DriverError> {
        let device = Device::new(0)?;
        let stream = &device.stream;
        let n = data.space.len();
        let real = |values: &[T]| values.iter().map(|v| v.re()).collect::<Vec<f64>>();
        let index = |values: Vec<usize>| values.into_iter().map(|v| v as u64).collect::<Vec<u64>>();

        let mut blocks = Vec::with_capacity(one_body.blocks.len());
        let mut nscratch = 0;
        for block in &one_body.blocks {
            match block {
                OneBodyBlock::Factorised(block) => {
                    let source = &one_body.spin.parents[block.source_parent];
                    let target = &one_body.spin.parents[block.target_parent];
                    let (sa, fa, sb, fb) = block.factors.factors();

                    // The first spin is contracted first: alpha for alpha-first pairs.
                    let alpha = matches!(block.contraction, OneBodyContraction::AFirst);
                    let ((f1, s1, n_s1, n_t1), (f2, s2, n_s2)) = if alpha {
                        ((fa, sa, block.nsa, block.nta), (fb, sb, block.nsb))
                    } else {
                        ((fb, sb, block.nsb, block.ntb), (fa, sa, block.nsa))
                    };
                    nscratch = nscratch.max(n_t1 * n_s2);

                    // Sources grouped by their second-spin string, with their first-spin strings.
                    let split = |a: usize, b: usize| if alpha { (a, b) } else { (b, a) };
                    let mut grouped = vec![Vec::new(); n_s2];
                    for entry in &source.entries {
                        let (first, second) = split(entry.a, entry.b);
                        grouped[second].push((first, entry.det));
                    }

                    let mut offsets = vec![0usize];
                    let (mut first, mut sources) = (Vec::new(), Vec::new());
                    for group in &grouped {
                        for &(s1, det) in group {
                            first.push(s1);
                            sources.push(det);
                        }
                        offsets.push(first.len());
                    }

                    // Targets with their first- and second-spin strings.
                    let (mut targets, mut t1, mut t2) = (Vec::new(), Vec::new(), Vec::new());
                    for entry in &target.entries {
                        let (first, second) = split(entry.a, entry.b);
                        targets.push(entry.det);
                        t1.push(first);
                        t2.push(second);
                    }

                    blocks.push(DeviceOneBodyBlock::Factorised(Box::new(
                        DeviceFactorisedOneBodyBlock {
                            f1: stream.clone_htod(&real(f1))?,
                            s1: stream.clone_htod(&real(s1))?,
                            f2: stream.clone_htod(&real(f2))?,
                            s2: stream.clone_htod(&real(s2))?,
                            n_s1: n_s1 as u64,
                            n_t1: n_t1 as u64,
                            n_s2: n_s2 as u64,
                            offsets: stream.clone_htod(&index(offsets))?,
                            first: stream.clone_htod(&index(first))?,
                            sources: stream.clone_htod(&index(sources))?,
                            ntarget: targets.len() as u64,
                            targets: stream.clone_htod(&index(targets))?,
                            t1: stream.clone_htod(&index(t1))?,
                            t2: stream.clone_htod(&index(t2))?,
                        },
                    )));
                }
                OneBodyBlock::Orthogonal(block) => {
                    let parent = &one_body.spin.parents[block.parent];
                    let cache = &fock.fock_mocache[block.parent];

                    let mut single_offsets = vec![0usize];
                    let (mut single_source, mut single_f) = (Vec::new(), Vec::new());
                    let mut same_offsets = vec![0usize];
                    let (mut same_source, mut same_f, mut same_s) =
                        (Vec::new(), Vec::new(), Vec::new());
                    let mut targets = Vec::with_capacity(parent.entries.len());

                    for (k, entry) in parent.entries.iter().enumerate() {
                        let target = NOCIIndex(entry.det);
                        targets.push(entry.det);

                        // The tabulated single-excitation couplings of this row.
                        for &(source, f) in &block.singles[block.starts[k]..block.starts[k + 1]] {
                            single_source.push(source);
                            single_f.push(f.re());
                        }
                        single_offsets.push(single_source.len());

                        // Sources sharing this row's spin strings, with their `F` and `S` elements.
                        let state = data.space.spin_string_pair(target);
                        let group = &block.groups[block.opos[&(state.aid, state.bid)]];
                        for source in &group.targets {
                            let s = NOCIIndex(source.det);
                            same_source.push(source.det);
                            same_f.push(
                                calculate_f_pair_orthogonal(cache, data.space, target, s).re(),
                            );
                            same_s.push(calculate_s_pair_orthogonal(data.space, target, s).re());
                        }
                        same_offsets.push(same_source.len());
                    }

                    blocks.push(DeviceOneBodyBlock::Orthogonal(Box::new(
                        DeviceOrthogonalOneBodyBlock {
                            single_offsets: stream.clone_htod(&index(single_offsets))?,
                            single_source: stream.clone_htod(&index(single_source))?,
                            single_f: stream.clone_htod(&single_f)?,
                            same_offsets: stream.clone_htod(&index(same_offsets))?,
                            same_source: stream.clone_htod(&index(same_source))?,
                            same_f: stream.clone_htod(&same_f)?,
                            same_s: stream.clone_htod(&same_s)?,
                            ntarget: targets.len() as u64,
                            targets: stream.clone_htod(&index(targets))?,
                        },
                    )));
                }
                OneBodyBlock::Transient(_) | OneBodyBlock::Contracted(_) => {
                    unreachable!("only factorised and orthogonal pairs are uploaded")
                }
            }
        }

        let module = device.load(KERNELS)?;

        Ok(Self {
            first: module.load_function("apply_one_body_factorised_first")?,
            second: module.load_function("apply_one_body_factorised_second")?,
            orthogonal: module.load_function("apply_one_body_orthogonal")?,
            device,
            blocks,
            n,
            nscratch,
        })
    }

    /// Allocate the first-contraction buffers, the counterpart of `OneBodyFactorisation::scratch`.
    /// # Arguments:
    /// - `self`: Device operator.
    /// # Returns:
    /// - `Result<DeviceOneBodyScratch, DriverError>`: Zeroed buffers.
    pub(super) fn scratch(&self) -> Result<DeviceOneBodyScratch, DriverError> {
        let stream = &self.device.stream;

        Ok(DeviceOneBodyScratch {
            first_f: stream.alloc_zeros(self.nscratch.max(1))?,
            first_s: stream.alloc_zeros(self.nscratch.max(1))?,
        })
    }

    /// Apply `y = (F + \lambda S) x` on the GPU, the counterpart of
    /// `OneBodyFactorisation::apply_one_body`.
    /// # Arguments:
    /// - `self`: Device operator.
    /// - `x`: Source determinant vector on the GPU.
    /// - `lambda`: Overlap shift `\lambda`.
    /// - `scratch`: First-contraction buffers.
    /// # Returns:
    /// - `Result<CudaSlice<f64>, DriverError>`: The product `y` on the GPU.
    pub(crate) fn apply_one_body(
        &self,
        x: &CudaSlice<f64>,
        lambda: f64,
        scratch: &mut DeviceOneBodyScratch,
    ) -> Result<CudaSlice<f64>, DriverError> {
        let mut y = self.device.stream.alloc_zeros(self.n)?;

        // Launches run in order on one stream, so the pairs' additions into `y` never overlap.
        for block in &self.blocks {
            match block {
                DeviceOneBodyBlock::Factorised(block) => {
                    self.apply_one_body_factorised(block, (x, &mut y), lambda, scratch)?
                }
                DeviceOneBodyBlock::Orthogonal(block) => {
                    self.apply_one_body_orthogonal(block, (x, &mut y), lambda)?
                }
            }
        }

        Ok(y)
    }

    /// Apply one factorised parent pair, the counterpart of
    /// `OneBodyFactorisation::apply_one_body_factorised`: the first contractions `T^F = F_1 D`
    /// and `T^S = S_1 D`, then the second contraction added into `y`, one warp per target.
    /// # Arguments:
    /// - `self`: Device operator.
    /// - `block`: Factorised parent pair.
    /// - `vectors`: Source vector `x` and the product `y` being accumulated.
    /// - `lambda`: Overlap shift `\lambda`.
    /// - `scratch`: First-contraction buffers.
    /// # Returns:
    /// - `Result<(), DriverError>`: Adds this pair's contribution into `y`.
    fn apply_one_body_factorised(
        &self,
        block: &DeviceFactorisedOneBodyBlock,
        vectors: (&CudaSlice<f64>, &mut CudaSlice<f64>),
        lambda: f64,
        scratch: &mut DeviceOneBodyScratch,
    ) -> Result<(), DriverError> {
        let (x, y) = vectors;
        let stream = &self.device.stream;

        let mut launch = stream.launch_builder(&self.first);
        launch.arg(&block.f1).arg(&block.s1).arg(&block.n_s1);
        launch
            .arg(&block.offsets)
            .arg(&block.first)
            .arg(&block.sources);
        launch.arg(x).arg(&block.n_t1).arg(&block.n_s2);
        launch.arg(&mut scratch.first_f).arg(&mut scratch.first_s);
        let threads = (block.n_t1 * block.n_s2).max(1) as u32;
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }?;

        let mut launch = stream.launch_builder(&self.second);
        launch.arg(&scratch.first_f).arg(&scratch.first_s);
        launch.arg(&block.f2).arg(&block.s2).arg(&block.n_s2);
        launch
            .arg(&block.targets)
            .arg(&block.t1)
            .arg(&block.t2)
            .arg(&block.ntarget);
        launch.arg(&lambda).arg(y);
        let threads = (WARP * block.ntarget).max(1) as u32;
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }?;

        Ok(())
    }

    /// Apply one same-parent orthogonal pair, the counterpart of
    /// `OneBodyFactorisation::apply_one_body_orthogonal`, one thread per target.
    /// # Arguments:
    /// - `self`: Device operator.
    /// - `block`: Same-parent orthogonal pair.
    /// - `vectors`: Source vector `x` and the product `y` being accumulated.
    /// - `lambda`: Overlap shift `\lambda`.
    /// # Returns:
    /// - `Result<(), DriverError>`: Adds this pair's contribution into `y`.
    fn apply_one_body_orthogonal(
        &self,
        block: &DeviceOrthogonalOneBodyBlock,
        vectors: (&CudaSlice<f64>, &mut CudaSlice<f64>),
        lambda: f64,
    ) -> Result<(), DriverError> {
        let (x, y) = vectors;
        let stream = &self.device.stream;

        let mut launch = stream.launch_builder(&self.orthogonal);
        launch
            .arg(&block.single_offsets)
            .arg(&block.single_source)
            .arg(&block.single_f);
        launch.arg(&block.same_offsets).arg(&block.same_source);
        launch.arg(&block.same_f).arg(&block.same_s);
        launch.arg(&block.targets).arg(&block.ntarget).arg(x);
        launch.arg(&lambda).arg(y);
        let threads = block.ntarget.max(1) as u32;
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }?;

        Ok(())
    }
}
