module JutulKLUExt
using Jutul
using SparseArrays: SparseMatrixCSC, nnz
import KLU

Jutul.check_klu_availability_impl() = true

Jutul.factorize_jacobian(::Val{:klu}, J::SparseMatrixCSC) = KLU.klu(J)
Jutul.refactorize_jacobian!(F::KLU.KLUFactorization, J) = KLU.klu!(F, J)
Jutul.lu_backend(::KLU.KLUFactorization) = :klu
Jutul.lu_fill_ratio(F::KLU.KLUFactorization, J) = (F.lnz + F.unz) / nnz(J)

end
