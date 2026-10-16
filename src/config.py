from pathlib import Path

#1.获取文件所在根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#2.判断文件目录是否存在
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESS_DIR = PROJECT_ROOT / "data" / "process"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LLM_DIR = PROJECT_ROOT / "data" / "llm"
LLM_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = PROJECT_ROOT / "data" / "tmp"
#3.自定义参数
DEFAULT_ADJUST = "hfq"    # 默认后复权（保证数据真实性）
REQUEST_SLEEP  = 1.0      # AKShare 限制访问间隔
CACHE_ENABLED  = True     # 开启缓存功能（避免频繁I/O、大模型请求和节省资源）     
N_SPLITS = 5
EMBARGO = 5