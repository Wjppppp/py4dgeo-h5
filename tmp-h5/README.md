# py4dgeo SpatiotemporalAnalysis HDF5 variant

## Todo

1. Implementing the converter between an analysis object and a hdf5 file
    For example:

    ```python

    from py4dgeo.io import analysis_to_hdf5, analysis_from_hdf5

    analysis = SpatiotemporalAnalysis("analysis.zip")

    analysis_to_hdf5(
        analysis, # or "analysis.zip",
        "analysis.h5",
        ...,
    )

    analysis_from_hdf5(
        "analysis.h5",
        "restored_analysis.zip",
        ...,
    )

    ```
2. Cross-valid elements in the analysis objects, and ensuring query efficiency (designing some bechmarks)
3. Preparing a notebook demo, following [Contributing Guidelines](https://github.com/Wjppppp/py4dgeo-h5/blob/main/CONTRIBUTING.md)
