(::TruePositive)(ft::FairTensor) = sum(ft.mat[:, 1, 1])
