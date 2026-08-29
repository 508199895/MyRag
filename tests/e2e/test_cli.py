from src.service import RagService


def test_run_interactive_reprompts_empty_input_then_exits(monkeypatch, capsys):
    service = RagService.__new__(RagService)
    service.ask = lambda question: f"answer:{question}"
    inputs = iter(["", "红烧肉怎么做", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    service.run_interactive()

    output = capsys.readouterr().out
    assert "您的问题是：" in output
    assert "请输入问题" in output
    assert "answer:红烧肉怎么做" in output
