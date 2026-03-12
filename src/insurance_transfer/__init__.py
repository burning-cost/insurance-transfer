"""
insurance-transfer: Transfer learning for thin-segment insurance pricing.

Implements covariate shift diagnostics, penalised GLM transfer (Tian & Feng, JASA 2023),
GBM source-as-offset transfer (CatBoost), and CANN pre-train/fine-tune (PyTorch optional).

References:
    Tian, Y. and Feng, Y. (2023). Transfer Learning under High-Dimensional Generalized
    Linear Models. Journal of the American Statistical Association, 118(544), 2684-2697.

    Loke, S.-H. and Bauer, D. (2025). Transfer Learning in the Actuarial Domain:
    Foundations and Applications. North American Actuarial Journal.
    DOI: 10.1080/10920277.2025.2489637.
"""

from insurance_transfer.shift import CovariateShiftTest, ShiftTestResult
from insurance_transfer.glm_transfer import GLMTransfer
from insurance_transfer.gbm_transfer import GBMTransfer
from insurance_transfer.diagnostic import NegativeTransferDiagnostic, TransferDiagnosticResult
from insurance_transfer.pipeline import TransferPipeline

__version__ = "0.1.0"
__all__ = [
    "CovariateShiftTest",
    "ShiftTestResult",
    "GLMTransfer",
    "GBMTransfer",
    "NegativeTransferDiagnostic",
    "TransferDiagnosticResult",
    "TransferPipeline",
]
