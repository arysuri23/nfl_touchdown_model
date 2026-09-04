import sys
from pathlib import Path

# Insert the wr_te/ directory (parent of tests/) at the front of sys.path
# so `import config` works regardless of the invocation cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))
