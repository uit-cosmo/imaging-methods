import imaging_methods as im


def get_all_velocities(N, simulation_data_getter, method_parameters):
    """
    Run N realisations and return the *raw* velocity components.

    simulation_data_getter: a callable function that for a given integer i returns the ith simulation data
    """
    v_2dca_all = []
    w_2dca_all = []
    v_2dcc_all = []
    w_2dcc_all = []
    v_2dca_max_all = []
    w_2dca_max_all = []
    v_2dcc_max_all = []
    w_2dcc_max_all = []
    vxtde_all = []
    vytde_all = []

    for i in range(N):
        ds = simulation_data_getter(i)
        v_input = ds["v_input"]
        w_input = ds["w_input"]

        (
            v_2dca,
            w_2dca,
            v_2dcc,
            w_2dcc,
            v_2dca_max,
            w_2dca_max,
            v_2dcc_max,
            w_2dcc_max,
            vxtde,
            vytde,
        ) = im.estimate_velocities_synthetic_ds(ds, method_parameters)

        v_2dca_all.append(v_2dca - v_input)
        w_2dca_all.append(w_2dca - w_input)
        v_2dcc_all.append(v_2dcc - v_input)
        w_2dcc_all.append(w_2dcc - w_input)
        v_2dca_max_all.append(v_2dca_max - v_input)
        w_2dca_max_all.append(w_2dca_max - w_input)
        v_2dcc_max_all.append(v_2dcc_max - v_input)
        w_2dcc_max_all.append(w_2dcc_max - w_input)
        vxtde_all.append(vxtde - v_input)
        vytde_all.append(vytde - w_input)

    return (
        v_2dca_all,
        w_2dca_all,
        v_2dcc_all,
        w_2dcc_all,
        v_2dca_max_all,
        w_2dca_max_all,
        v_2dcc_max_all,
        w_2dcc_max_all,
        vxtde_all,
        vytde_all,
    )
