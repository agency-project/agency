import json

from agency.native_harness import libsuggest

SETTINGS = "\n".join(["import django", "from django.conf import settings",
                      "settings.configure(DEBUG=True, INSTALLED_APPS=['django.contrib.admin', 'django.contrib.auth',",
                      "    'django.contrib.contenttypes', 'django.contrib.sessions', 'django.contrib.messages'],",
                      "    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},",
                      "    ROOT_URLCONF='urls', SECRET_KEY='x', MIDDLEWARE=[], TEMPLATES=[])", "django.setup()"])
JSONF = "from django.db.models import JSONField\nprint(JSONField().get_prep_value({'a': 1}))\n"


def _lib(tmp_path, monkeypatch):
    (tmp_path / "lib_001.py").write_text(SETTINGS)
    (tmp_path / "lib_002.py").write_text(JSONF)
    (tmp_path / "INDEX.md").write_text(
        "# Script library\n\n| file | purpose | written in | runs |\n|---|---|---|---|\n"
        "| lib_001.py | Minimal Django settings for testing admin inlines. | D7-django-11095 | 2 |\n"
        "| lib_002.py | Reproduces JSONField has_key lookups with numeric keys. | B1-django-15503 | 5 |\n")
    monkeypatch.setattr(libsuggest, "LIB_DIR", str(tmp_path))
    libsuggest.reset()


def test_start_note_ranks_scripts_by_the_task_text(tmp_path, monkeypatch):
    _lib(tmp_path, monkeypatch)
    note = libsuggest.start_note("JSONField has_key lookup fails for numeric string keys on sqlite")
    assert note.startswith("[harness: script library]")
    lines = [l for l in note.splitlines() if l.startswith("  ")]
    assert "common setup boilerplate" in lines[0] and "lib_001.py" in lines[0]
    assert "lib_002.py" in lines[1]


def test_write_note_fires_once_when_a_library_script_is_recreated(tmp_path, monkeypatch):
    _lib(tmp_path, monkeypatch)
    args = {"command": f"cat > /tmp/settings_check.py <<'EOF'\n{SETTINGS}\nprint('ok')\nEOF\npython /tmp/settings_check.py"}
    handler = libsuggest.with_note("bash", lambda a: json.dumps({"output": "ok", "returncode": 0}))
    out = json.loads(handler(json.dumps(args)))["output"]
    assert "re-creates an existing library script" in out and "lib_001.py" in out
    assert "[harness" not in json.loads(handler(json.dumps(args)))["output"]


def test_unrelated_scripts_and_library_paths_get_no_note(tmp_path, monkeypatch):
    _lib(tmp_path, monkeypatch)
    assert libsuggest.write_note("write", {"file_path": "/tmp/x.py", "content": "print('hello world')\n" * 3}) == ""
    assert libsuggest.write_note("write", {"file_path": f"{tmp_path}/lib_009.py", "content": SETTINGS}) == ""


def test_helper_hint_fires_once_when_a_script_does_what_a_helper_does(tmp_path, monkeypatch):
    _lib(tmp_path, monkeypatch)
    (tmp_path / "helpers.json").write_text(json.dumps(
        [{"pattern": r"settings\.configure\(", "note": "use h3lib.django_env.setup()"}]))
    libsuggest.reset()
    args = {"file_path": "/tmp/s.py", "content": "from django.conf import settings\nsettings.configure(DEBUG=True)\n"}
    assert libsuggest.write_note("write", args) == "[harness: script library] use h3lib.django_env.setup()"
    assert "h3lib" not in libsuggest.write_note("write", args)
    assert "importable helpers" in libsuggest.start_note("django settings configure")
