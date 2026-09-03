"""Truth-blind page grouping for incoming claim packages."""

from .contracts import AssembledDocument, AssemblyResult, PageSignal
from .service import DocumentAssemblyService

__all__ = ["AssembledDocument", "AssemblyResult", "PageSignal", "DocumentAssemblyService"]
