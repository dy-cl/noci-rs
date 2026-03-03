// write.rs
use hdf5::File;
use hdf5::types::VarLenUnicode;

use ndarray::{Array1, Array2};

use crate::AoData;

/// Write sufficient data of an SCF state into an HDF5  file such that we can plot the orbitals in post. 
/// Of course, this does not contain the geometry or basis used, but this can be provided in the input to a
/// plotting script provided we have not forgotten what was used.
/// # Arguments:
///     `path`: String, filepath for the HDF5 file.
///     `label`: String, sanitised label of the SCF state.
///     `ao`: AoData struct, contains AO integrals and metadata.
///     `ca`: Array2, MO coefficients spin alpha.  
///     `cb`: Array2, MO coefficients spin beta.
///     `ea`: Array1, MO energies spin alpha. 
///     `eb`: Array1, MO energies spin beta.
///     `oa`: Array1, MO occupancies spin alpha.
///     `ob`: Array1, MO occupancies spin beta.
///     `da`: Array2, spin alpha density matrix.
///     `db`: Array2, spin beta density matrix.
pub fn write_orbitals(path: &str, ao: &AoData, label: &str, ca: &Array2<f64>, cb: &Array2<f64>, ea: &Array1<f64>, eb: &Array1<f64>,
                      oa: &Array1<f64>, ob: &Array1<f64>, da: &Array2<f64>, db: &Array2<f64>) {
    let f = File::create(path).unwrap();
    
    let vlabel: VarLenUnicode = label.parse().unwrap();
    f.new_dataset::<VarLenUnicode>().create("label").unwrap().write_scalar(&vlabel).unwrap();
    let labelsvlu: Vec<VarLenUnicode> = ao.labels.iter().map(|s| s.parse().unwrap()).collect();
    f.new_dataset::<VarLenUnicode>().shape(labelsvlu.len()).create("aolabels").unwrap().write(&labelsvlu).unwrap();

    let neleci64: Vec<i64> = ao.nelec.iter().copied().collect();
    f.new_dataset::<i64>().shape(neleci64.len()).create("nelec").unwrap().write(&neleci64).unwrap();
    
    let n = ao.s.ncols();
    f.new_dataset::<f64>().shape((n, n)).create("S").unwrap().write(&ao.s).unwrap();
    f.new_dataset::<f64>().shape((n, n)).create("ca").unwrap().write(ca).unwrap();
    f.new_dataset::<f64>().shape((n, n)).create("cb").unwrap().write(cb).unwrap();
    f.new_dataset::<f64>().shape((n, n)).create("da").unwrap().write(da).unwrap();
    f.new_dataset::<f64>().shape((n, n)).create("db").unwrap().write(db).unwrap();
    f.new_dataset::<f64>().shape(ea.len()).create("ea").unwrap().write(ea).unwrap();
    f.new_dataset::<f64>().shape(eb.len()).create("eb").unwrap().write(eb).unwrap();
    f.new_dataset::<f64>().shape(oa.len()).create("oa").unwrap().write(oa).unwrap();
    f.new_dataset::<f64>().shape(ob.len()).create("ob").unwrap().write(ob).unwrap();
}
