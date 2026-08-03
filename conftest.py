import os
import sys

# 저장소 루트를 import 경로에 넣어 `import ofi_spec`, `from preprocessing... `
# 가 테스트에서도 그대로 동작하게 한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
