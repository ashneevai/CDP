from packages.document_assembly import DocumentAssemblyService, PageSignal


def test_class_change_creates_logical_document_boundary():
    service = DocumentAssemblyService(minimum_confidence=0.65)
    result = service.assemble([
        PageSignal("p1", 1, "CMS1500", 0.98),
        PageSignal("p2", 2, "CMS1500", 0.91, continuation_hint=True),
        PageSignal("p3", 3, "EOB", 0.95),
    ])
    assert [document.document_class for document in result.documents] == ["CMS1500", "EOB"]
    assert result.documents[0].page_ids == ("p1", "p2")


def test_uncertain_page_is_not_force_grouped():
    service = DocumentAssemblyService(minimum_confidence=0.70)
    result = service.assemble([
        PageSignal("p1", 1, "UB04", 0.95),
        PageSignal("p2", 2, "UB04", 0.40),
        PageSignal("p3", 3, "UB04", 0.92),
    ])
    assert result.uncertain_page_ids == ("p2",)
    assert len(result.documents) == 2


def test_blank_and_duplicate_pages_are_explicitly_ignored():
    result = DocumentAssemblyService().assemble([
        PageSignal("blank", 1, "UNKNOWN", 0.99, is_blank=True),
        PageSignal("dup", 2, "CMS1500", 0.99, is_duplicate=True),
        PageSignal("claim", 3, "CMS1500", 0.99),
    ])
    assert result.ignored_page_ids == ("blank", "dup")
    assert result.documents[0].page_ids == ("claim",)


def test_assembly_service_has_no_claim_or_field_decision_authority():
    service = DocumentAssemblyService()
    assert not hasattr(service, "decide")
    assert not hasattr(service, "accept")
