from packages.document_taxonomy.taxonomy import DocumentClass

from .cms1500 import _result
from .evidence import StandardFormEvidence


class UB04Verifier:
    policy_version = "ub04-verifier-v2"

    def verify(self, evidence: StandardFormEvidence):
        if evidence.candidate_family != DocumentClass.UB04:
            raise ValueError("UB04Verifier requires a UB04 nomination")
        return _result(evidence, self.policy_version)