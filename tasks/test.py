def test_openai_api():
	from openai import OpenAI
	import os
	# from finch import Finch

	# fin = Finch(metric='cosine')
	exit(0)

	# proxy_url = "http://host.docker.internal:7890" 

	# os.environ["HTTP_PROXY"] = proxy_url
	# os.environ["HTTPS_PROXY"] = proxy_url

	# --- 严重安全警告! ---
	# 请不要像下面这样直接在代码中写入密钥 (hardcode)。
	# 这非常不安全。这里只是为了演示您的请求。
	# 测试后请立即到您的 API 供应商后台删除此密钥！
	# API_KEY = "hk-10t3l5100005993509f283dc8f4a81a4a55e3d3e191ee8d4"
	# BASE_URL = "https://api.openai-hk.com/v1"
	# model="gpt-5-mini"


	API_KEY = "TDM6IgMVUcG9sfHeweMMgrUD4ptayo8J"
	BASE_URL = "https://antchat.alipay.com/v1"
	model = "Qwen3-235B-A22B-Instruct-2507"

	# curl https://antchat.alipay.com/v1

	# ----------------------------------------------------
	# 推荐的、更安全的做法是使用环境变量
	# （您需要先在系统中设置它们）
	# 
	# from dotenv import load_dotenv
	# load_dotenv() # 需要先 pip install python-dotenv
	# API_KEY = os.environ.get("MY_HK_API_KEY")
	# BASE_URL = os.environ.get("MY_HK_BASE_URL")
	# ----------------------------------------------------


	try:
			# 初始化 OpenAI 客户端
			# 客户端会使用您提供的 base_url 和 api_key
			client = OpenAI(
					api_key=API_KEY,
					base_url=BASE_URL
			)

			print(f"正在尝试连接到: {BASE_URL}...")

			# 发起一个简单的聊天补全 (Chat Completion) 请求
			# 这是测试 API 是否工作最常用的方法
			completion = client.chat.completions.create(
				model=model,  # 您可以根据您的 API 平台支持的模型更改此名称
				messages=[
					{"role": "system", "content": "You are a helpful assistant."},
					{"role": "user", "content": "你好，请回复 '测试成功'"}
				]
			)

			print("\n✅ API 调用成功！")
			print("="*30)
			# 打印来自模型的回复内容
			print("模型回复:", completion.choices[0].message.content)
			print("="*30)
			
			# (可选) 打印完整的响应对象以供调试
			# print("\n完整响应 (调试用):")
			# print(completion)

	except Exception as e:
			print(f"\n❌ API 调用失败。")
			print(f"错误类型: {type(e).__name__}")
			print(f"错误详情: {e}")

def test_chroma():
	"""
	"""
	import os
	import shutil
	from langchain_chroma import Chroma
	from langchain_community.embeddings import HuggingFaceEmbeddings
	from langchain.docstore.document import Document


	# --- 配置 ---
	# 1. 定义持久化目录
	PERSIST_DIR = "/Users/liuyi/Documents/Docker_env/Agent/GMemory-main/tasks/test_out_data/chroma_persistence_test_db"

	# 2. 定义要使用的嵌入模型（一个轻量级的本地模型）
	#    HuggingFaceEmbeddings 会自动下载并使用它
	EMBED_MODEL = "sentence-transformers/all-miniLM-L6-v2"

	# 3. 准备嵌入函数
	print("正在初始化嵌入模型...")
	embedding_func = HuggingFaceEmbeddings(
			model_name=EMBED_MODEL,
			model_kwargs={'device': 'cpu'} # 强制使用 CPU，如果没 GPU 的话
	)
	# 4. 准备要存储的数据
	documents_to_add = [
			Document(page_content="Chroma 是一个向量数据库。", metadata={"source": "doc1"}),
			Document(page_content="它支持数据的持久化存储。", metadata={"source": "doc2"}),
	]
	# --- 核心逻辑：检查是第一次还是第二次运行 ---

	db = None

	if not os.path.exists(PERSIST_DIR):
			# --- 场景一：第一次运行 (目录不存在) ---
			print("\n" + "="*30)
			print(f"*** 第一次运行：未找到目录 {PERSIST_DIR} ***")
			print("="*30)
			print("正在创建新的数据库并添加 2 个文档...")

			# 1. 初始化：
			#    这会创建空的数据库目录和文件结构
			#    (注意：我们不再使用 .from_documents)
			print(f"正在 {PERSIST_DIR} 初始化一个空的数据库...")
			db = Chroma(
					persist_directory=PERSIST_DIR,
					embedding_function=embedding_func
			)
			# 2. 添加：
			#    现在，我们显式调用 .add_documents() 来添加内容
			print("数据库已空置初始化。现在调用 .add_documents() 添加 2 个文档...")
			db.add_documents(documents=documents_to_add)
			
			print("文档添加完成。")
			# 注意：由于我们在初始化时就指定了 persist_directory，
			# add_documents() 会自动将更改持久化到磁盘。
			# (如果初始化时未指定目录，db.persist() 才需要手动调用)

	else:
			# --- 场景二：第二次运行 (目录已存在) ---
			print("\n" + "="*30)
			print(f"*** 第二次运行：发现已存在的目录 {PERSIST_DIR} ***")
			print("="*30)
			print("正在加载现有数据库...")
			
			# 这就是您问题中的代码：
			# 它会直接加载 PERSIST_DIR 中的数据
			db = Chroma(
					persist_directory=PERSIST_DIR,
					embedding_function=embedding_func
			)
			print("数据库加载完成。")


	# --- 通用操作：无论第几次运行，都执行检索 ---

	print("\n" + "-"*30)
	print("开始执行数据库操作...")
	print("-" * 30)

	# 操作 1: 检索 (Similarity Search)
	query = "什么是 Chroma？"
	print(f"\n[操作 1] 正在执行相似性检索，查询: '{query}'")
	results = db.similarity_search(query, k=1)

	if results:
			print("检索结果:")
			print(f"  - 内容: {results[0].page_content}")
			print(f"  - 元数据: {results[0].metadata}")
	else:
			print("未找到相关结果。")


	# 操作 2: 获取所有内容 (get)
	print("\n[操作 2] 正在获取数据库中的所有文档...")
	all_documents_data = db.get() # .get() 返回一个包含 ids, embeddings, documents, metadatas 的字典

	doc_count = len(all_documents_data.get('ids', []))
	print(f"数据库中的文档总数: {doc_count}")

	if doc_count > 0:
			print("所有文档内容:")
			for doc in all_documents_data.get('documents', []):
					print(f"  - {doc}")

	# --- 结论 ---
	print("\n" + "="*30)
	if doc_count == 2:
			print("测试成功！数据库内容已成功继承。")
	else:
			print("测试失败或数据库为空。")
	print("="*30)


	# --- (可选) 清理 ---
	# 如果你想重新开始“第一次运行”，取消下面这行代码的注释
	# print(f"\n如需重置，请手动删除目录: shutil.rmtree('{PERSIST_DIR}')")
	# shutil.rmtree(PERSIST_DIR) 
	# print("清理完成。")

if __name__== "__main__":
		test_chroma()
		print()
