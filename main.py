import Classes.EigenProblemClass as EP

eigen_problem = EP.FENicSEigenProblem(
    num_nodes=50, domain_type="cube", test_mode=False, num_eigenvalues=25
)

eigen_problem.run()
