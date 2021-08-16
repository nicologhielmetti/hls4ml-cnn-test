


v_master_cosim = np.array(parse_sim_log('V@master/V@master-cosim.log', 'Vivado'))
v_master_csim = np.array(parse_sim_log('V@master/V@master-csim.log', 'Vivado'))

v_pr_cosim = np.array(parse_sim_log('V@PR/V@PR-cosim.log', 'Vivado'))
v_pr_csim = np.array(parse_sim_log('V@PR/V@PR-csim.log', 'Vivado'))

va_cosim = np.array(parse_sim_log('VA/VA-cosim.log', 'VivadoAccelerator'))
va_csim = np.array(parse_sim_log('VA/VA-csim.log', 'VivadoAccelerator'))
va_hw = np.load('VA/VA-HW.npy').ravel()

try:

    # --- 1st row ---

    print(' va_csim vs va_cosim')

    np.testing.assert_array_equal(x=va_csim, y=va_cosim)

    print(' va_csim vs va_hw')

    np.testing.assert_array_equal(x=va_csim, y=va_hw)

    print(' va_csim vs v_pr_csim')

    np.testing.assert_array_equal(x=va_csim, y=v_pr_csim)

    print(' va_csim vs v_pr_cosim')

    np.testing.assert_array_equal(x=va_csim, y=v_pr_cosim)

    print(' va_csim vs v_master_csim')

    np.testing.assert_array_equal(x=va_csim, y=v_master_csim)

    print(' va_csim vs v_master_cosim')

    np.testing.assert_array_equal(x=va_csim, y=v_master_cosim)

    # --- 2nd row ---

    print(' va_cosim vs va_hw')

    np.testing.assert_array_equal(x=va_cosim, y=va_hw)

    print(' va_cosim vs v_pr_csim')

    np.testing.assert_array_equal(x=va_cosim, y=v_pr_csim)

    print(' va_cosim vs v_pr_cosim')

    np.testing.assert_array_equal(x=va_cosim, y=v_pr_cosim)

    print(' va_cosim vs v_master_csim')

    np.testing.assert_array_equal(x=va_cosim, y=v_master_csim)

    print(' va_cosim vs v_master_cosim')

    np.testing.assert_array_equal(x=va_cosim, y=v_master_cosim)

    # --- 3rd row ---

    print(' va_hw vs v_pr_csim')

    np.testing.assert_array_equal(x=va_hw, y=v_pr_csim)

    print(' va_hw vs v_pr_cosim')

    np.testing.assert_array_equal(x=va_hw, y=v_pr_cosim)

    print(' va_hw vs v_master_csim')

    np.testing.assert_array_equal(x=va_hw, y=v_master_csim)

    print(' va_hw vs v_master_cosim')

    np.testing.assert_array_equal(x=va_hw, y=v_master_cosim)

    # --- 4th row ---

    print(' v_pr_csim vs v_pr_cosim')

    np.testing.assert_array_equal(x=v_pr_csim, y=v_pr_cosim)

    print(' v_pr_csim vs v_master_csim')

    np.testing.assert_array_equal(x=v_pr_csim, y=v_master_csim)

    print(' v_pr_csim vs v_master_cosim')

    np.testing.assert_array_equal(x=v_pr_csim, y=v_master_cosim)

    # --- 5th row ---

    print(' v_pr_cosim vs v_master_csim')

    np.testing.assert_array_equal(x=v_pr_cosim, y=v_master_csim)

    print(' v_pr_cosim vs v_master_cosim')

    np.testing.assert_array_equal(x=v_pr_cosim, y=v_master_cosim)

    # --- 6th row ---

    print(' v_master_csim vs v_master_cosim')

    np.testing.assert_array_equal(x=v_master_csim, y=v_master_cosim)
except Exception:
    pass
