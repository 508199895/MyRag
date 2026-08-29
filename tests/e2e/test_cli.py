from src.service import RagService


def test_run_interactive_reprompts_empty_input_then_exits(monkeypatch, capsys):
    service = RagService.__new__(RagService)
    asked_questions = []

    def ask(question):
        asked_questions.append(question)
        return f"answer:{question}"

    service.ask = ask
    inputs = iter(["", "   ", "红烧肉怎么做", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    service.run_interactive()

    output = capsys.readouterr().out
    assert "您的问题是：" in output
    assert output.count("请输入问题") == 2
    assert "answer:红烧肉怎么做" in output
    assert asked_questions == ["红烧肉怎么做"]


def test_run_interactive_strips_exit_command_before_asking(monkeypatch, capsys):
    service = RagService.__new__(RagService)
    asked_questions = []

    def ask(question):
        asked_questions.append(question)
        return f"answer:{question}"

    service.ask = ask
    inputs = iter(["   q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    service.run_interactive()

    assert "您的问题是：" in capsys.readouterr().out
    assert asked_questions == []
