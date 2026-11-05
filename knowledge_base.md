# 踩坑知识库（RAG 任务 7 示例）

下面每篇 `## ` 开头的都是一篇独立文档，RAG 会按这个切分。

## pip 安装时报 SSL 错误

现象：`pip install` 时出现 `SSLError: EOF occurred in violation of protocol`，重试几次也没用，连换了清华源也一样。
原因：公司网络或系统代理干扰了 TLS 握手，不是镜像源的问题。
解法（按顺序试）：
1. PowerShell 先跑 `$env:no_proxy="*"` 绕过系统代理，再 `pip install`。
2. 还不行就切手机热点装包，装完切回来。
3. 把清华源设为默认：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`，以后装包都快。

## git push 被拒绝（fetch first）

现象：`git push` 报 `! [rejected] main -> main (fetch first)`。
原因：远程仓库有你本地没有的提交（比如建仓库时 GitHub 自动生成的 README）。
解法：先 `git pull --rebase origin main` 把远端的拿下来，再 `git push`。如果确定远端的东西不要了（比如自己的个人仓库），可以用 `git push -u origin main -f` 强推覆盖。

## git 连不上 github.com:443

现象：push 时报 `Failed to connect to github.com:443`。
解法：先直接重试一次，瞬时抽风很常见；还不行跑 `git config --global http.version HTTP/1.1`；再不行切手机热点。

## Python 缩进报错 IndentationError

现象：运行报 IndentationError，多半是缩进里混了空格和 Tab，或者复制粘贴时缩进乱了。
解法：别用记事本写 Python，换 VS Code 并装微软 Python 插件；多处修改时直接换整份文件，别摞补丁。

## API key 不要写死在代码里

教训：曾经把带真实 key 的代码 push 到 GitHub，最后靠强推覆盖才清干净历史。
正确做法：key 存环境变量，代码里用 `os.getenv("DEEPSEEK_API_KEY")` 读取。Windows 上用 `setx DEEPSEEK_API_KEY "你的key"` 设置，新开终端生效。push 前全局搜 `sk-`，有结果就别推。

## HuggingFace 模型下载慢

现象：sentence-transformers 第一次加载模型，几百 MB 下不动。
解法：PowerShell 跑 `$env:HF_ENDPOINT="https://hf-mirror.com"`（同一窗口），再运行脚本，走国内镜像站。

## VS Code 里 Python 文件怎么运行

右上角 ▶ 直接在集成终端运行；或者在终端里 `cd` 到文件所在目录，跑 `python 文件名.py`。报错信息看 Terminal 面板，代码静态问题看 Problems 面板。

## embedding 是什么

一句话：把一句话变成一串数字（向量），意思越像，向量在空间里离得越近。
用处：RAG 找资料就靠它——查“意思”而不是查“关键词”。比如查“iPhone 屏幕碎了”能命中“苹果手机掉地上摔碎了”，尽管一个字都对不上。

## 删库重建 vs 强推覆盖

想清掉 Git 历史（比如误传了 key）有两种办法：
1. 删库重建：网页上删仓库 → 本地删 .git → 重新 init/add/commit → 新建空仓库 → push。步骤多，容易在删除确认那步卡住。
2. 强推覆盖：本地整理好干净的历史，直接 `git push -u origin main -f`，一次覆盖远端。个人仓库、无协作者时最省事。

## commit 信息怎么写

格式：一句话说清这次干了什么，比如“任务六：Embedding 初体验”。push 前先 `git status` 看一眼文件清单，确认没有多余文件，更不能有 key。

## Gradio share 链接建不起来（frpc 缺失）

现象：跑 `demo.launch(share=True)` 时本机地址正常，但建不了公开链接，报错 `Could not create share link. Missing file: ...\gradio\frpc\frpc_windows_amd64_v0.3`。
原因：公开链接要靠 frpc 这个隧道工具，Gradio 首次运行时自动下载它；当前网络下不下来，文件夹一直是空的。
解法（按顺序试）：
1. 浏览器手动下载 https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_windows_amd64.exe，改名成 `frpc_windows_amd64_v0.3`，移进 `C:\Users\<你>\.cache\huggingface\gradio\frpc`。
2. Windows 启动程序时名字没后缀会自动补 `.exe` 去找，所以文件夹里还要再复制一份 `frpc_windows_amd64_v0.3.exe`。
3. 注意：资源管理器改名时中文输入法可能混进全角字符，看着对、程序找不到；改名后用 `dir` 核对，找不到就用 `cmd /c "ren frpc* frpc_windows_amd64_v0.3"` 通配符改名。
4. 改完重启程序，公开链接就出来了。

## 网页版首次提问卡住（模型加载联网校验）

现象：Gradio 网页版第一次提问时一直 processing，几分钟都不出回答；终端里反复出现 `Retrying in 8s` 和 `WinError 10060`，请求的是 huggingface.co 的 HEAD。
原因：sentence-transformers 加载模型时默认先联网问一下 Hub（HEAD 校验），当前网络连不上 huggingface.co，就一次次超时重试；模型其实已经在本地缓存里了。
解法：启动前设 `$env:HF_HUB_OFFLINE="1"`，让它只读本地缓存、不联网问，再跑程序。注意这个变量只对当前终端有效，新开终端要重设。

## Hugging Face Spaces 的 Gradio 要付费

现象：新建 Space 时 Gradio 和 Docker 档位标着 Paid，提示要 PRO 订阅；免费账号只能建 Static。
原因：该账号下跑程序的 Space（Gradio/Docker）需要付费计划，免费的 Static 只能放静态网页，跑不了 Python 程序。
解法：要么订阅 PRO 继续用 Spaces；要么换免费的 Render（永久链接，闲置休眠、访问唤醒），Render 上程序要绑平台给的 PORT 端口；要么先用 gradio.live 临时链接（每次重启都换）。

## gradio.live 临时链接的规则

现象：公开链接这次能用，下次打开却显示 `No interface is running right now`。
原因：gradio.live 链接是临时的，每次启动程序都会换一个新链接；程序退出或电脑关机，隧道就没了，旧链接随即失效（官方说最长约 1 周）。
解法：要提问就保持程序开着；程序关了就重跑一次，拿终端里给的新链接。想要固定不变的链接，得部署到 Render、Spaces 这类平台。
