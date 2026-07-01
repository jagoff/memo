# vhs tape for docs/demo.gif — memo "save once, recall later" (~40s)
# Regenerate:  vhs docs/demo.tape   (installs: brew install vhs)
Output docs/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 640
Set Padding 24
Set Theme "Dracula"

# --- hidden setup: isolated store + fast default model (never touches real corpus) ---
Hide
Type 'export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"'
Enter
Type "export MEMO_DATA_DIR=/tmp/memo-demo MEMO_STATE_DIR=/tmp/memo-demo-state MEMO_NONINTERACTIVE=1 MEMO_MEMORIES_IN_VAULT=0"
Enter
Type "export MEMO_EMBEDDER_MODEL=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ MEMO_EMBEDDER_DIMS=1024"
Enter
Type "rm -rf /tmp/memo-demo /tmp/memo-demo-state && clear"
Enter
Show

Type "# memo: save a fact once — any later session recalls it automatically"
Enter
Sleep 1200ms
Type `memo save 'we use Postgres, not Mongo for the analytics service' --title 'db choice' -t decision`
Enter
Sleep 6s
Type "clear"
Enter
Sleep 400ms

Type "# --- a fresh session, later ---"
Enter
Sleep 900ms
Type `memo search 'what database did we pick for analytics'`
Enter
Sleep 5s
Sleep 2s
