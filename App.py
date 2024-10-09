import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import hashlib
import os
from streamlit_lottie import st_lottie
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient
import datetime
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import BaseMessage
from zhipuai import ZhipuAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter



#这里是各类变量, 和所用api的调用模块
_ = load_dotenv(find_dotenv())

load_dotenv()
api_key = "860779ea256ea8bc5b82bc4a80805346.5XOw3HFZ9ZZqjkxB"
tavily_answer = TavilyClient(api_key='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')
if not api_key:
    raise ValueError("ZHIPUAI_API_KEY is not set in the environment variables.")

# Initialize ZhipuAI client
zhipuai_client = ZhipuAI(api_key=api_key)
#ROLES_ALL_PROGRAM = ["🕵️‍♂️投喂宝", "👨‍🎓铲屎官", "👨‍🦯游客"]


# Function to interact with ZhipuAI and get response
def interact_with_zhipuai(question, usingTokens, tavily_Opened):
    if tavily_Opened == "Opened":
        if "血压" or "降压" or "压" not in question:
            Additional_Answer = "胖橘正在上网为您查找😾\n"
            qna_question = tavily_answer.qna_search(query=question)
            return Additional_Answer + qna_question
    # Create ChatZhipuAI object
    chat_zhipu_ai = ChatZhipuAI(
        api_key="860779ea256ea8bc5b82bc4a80805346.5XOw3HFZ9ZZqjkxB",
        model="glm-4",
        messages=[
            {"role": "user", "content": question}
        ],
        tools=[
            {
                "type": "retrieval",
                "retrieval": {
                    "knowledge_id": "1809964567753887744",
                    "prompt_template": (
                        "从文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n中找问题\n\"\"\"\n{{question}}\n\"\"\"\n"
                        "的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n"
                        "不要复述问题，直接开始回答。"
                    )
                }
            }
        ],
        stream=True,
        max_tokens=usingTokens

    )

    # Get response from ZhipuAI
    response = ''
    for chunk in zhipuai_client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": question}],
            tools=[
                {
                    "type": "retrieval",
                    "retrieval": {
                        "knowledge_id": "1809964567753887744",
                        "prompt_template": (
                                "从文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n中找问题\n\"\"\"\n{{question}}\n\"\"\"\n"
                                "的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n"
                                "不要复述问题，直接开始回答。"
                        )
                    }
                }
            ],
            stream=True,
            max_tokens=usingTokens
    ):
        response += chunk.choices[0].delta.content

    return response


# 胖橘记忆检索模块
def zhipu_chat_commemorate(question, temperatureChoice, promptAru):
    # str=input("Just feel free to ask me anything in the LangchainZhipuAI !")
    zhipuai_chat = ChatZhipuAI(
        temperature=temperatureChoice,
        api_key="860779ea256ea8bc5b82bc4a80805346.5XOw3HFZ9ZZqjkxB",
        model_name="glm-4",

    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '你是一个擅长{ability}的助手',
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    memory = ConversationBufferMemory(k=20)
    runnable = prompt | zhipuai_chat
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversation = RunnableWithMessageHistory(
        runnable=runnable,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    response = conversation.invoke(
        {"ability": "{}".format(promptAru), "input": question},
        config={"configurable": {"session_id": "abc123"}},
    )

    return response.content

#胖橘langgraph模块
memory = SqliteSaver.from_conn_string(":memory:")
zhipuai_client = ZhipuAI(api_key="860779ea256ea8bc5b82bc4a80805346.5XOw3HFZ9ZZqjkxB")
# 定义 State 类型
class State(TypedDict):
    messages: list[BaseMessage]  # 确保消息格式符合要求

graph_builder = StateGraph(State)

def askingfunction(question: str) -> dict:
    llm = ChatZhipuAI(
        model_name="glm-4",
        api_key="860779ea256ea8bc5b82bc4a80805346.5XOw3HFZ9ZZqjkxB",
        messages=[
            {"role": "user", "content": question}
        ],

        tools=[
            {
                "type": "retrieval",
                "retrieval": {
                    "knowledge_id": "1809964567753887744",
                    "prompt_template": (
                        "从文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n中找问题\n\"\"\"\n{{question}}\n\"\"\"\n"
                        "的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n"
                        "不要复述问题，直接开始回答。"
                    )
                }
            }
        ],
        stream=True,
        max_tokens=300
    )
    response = ''
    for chunk in zhipuai_client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": question}],
            tools=[
                {
                    "type": "retrieval",
                    "retrieval": {
                        "knowledge_id": "1809964567753887744",
                        "prompt_template": (
                                "从文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n中找问题\n\"\"\"\n{{question}}\n\"\"\"\n"
                                "的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n"
                                "不要复述问题，直接开始回答。"
                        )
                    }
                }
            ],
            stream=True,
            max_tokens=300
    ):
        response += chunk.choices[0].delta.content


    return {"messages": [response]}  # 确保返回格式是 dict


#GraphBuildingFunction
def chatBot(user_input: str) -> dict:
    # 直接使用用户输入
    return askingfunction(user_input)

graph_builder.add_node("chatBot", chatBot)
graph_builder.add_edge(START, "chatBot")
graph_builder.add_edge("chatBot", END)

graph = graph_builder.compile(checkpointer=memory)

#这是一个未图形化界面的测试用例
# 主循环接收用户输入'''
#while True:
#    user_input = input("User: ")
#    if user_input.lower() in ["quit", "exit", "q"]:
#        print("goodBye")
#        break
#    # 使用用户输入调用 chatBot 函数
#    response = chatBot(user_input)
#...




# Page 1: ZhipuAI RAG 智能检索

def page_zhipuai_rag(CatFeed, tavily_Opened):
    st.title(':whale:心脉AI RAG智能检索大模型')
    st.markdown(
        """
        <div style='height: 2px; background: linear-gradient(to right, #ff7e5f, #feb47b);'></div>
        """,
        unsafe_allow_html=True)

    # Initialize session state for storing chat history
    if 'messages_rag' not in st.session_state:
        st.session_state.messages_rag = []

    # Display chat history
    for role, message in st.session_state.messages_rag:
        with st.expander(role.capitalize()):
            st.write(message)

    # Input box for user question
    user_input = st.text_area(':grinning_face_with_one_large_and_one_small_eye:请将你的小鱼干放进去(问问吧):', '')

    # Button to trigger response
    if st.button('👩‍🚀获取答案！'):
        if user_input:
            # Add user message to chat history
            st.session_state.messages_rag.append(("🥰投喂人", user_input))

            # Get response from ZhipuAI
            response = interact_with_zhipuai(user_input, CatFeed, tavily_Opened)

            # Add ZhipuAI response to chat history
            st.session_state.messages_rag.append(("😾心脉AI", response))

        else:
            st.write('心脉AI没有检测到鱼干🐠')


# Page 2: 胖橘记忆检索

def page_zhipuai_memory(temperatureChoicePage, ChoosingPrompt):
    st.title(':penguin:心脉AI记忆检索大模型')

    st.markdown(
        """
        <div style='height: 20px; background: linear-gradient(to right, #ff7e5f, #feb47b);'></div>
        """,
        unsafe_allow_html=True)

    # Initialize session state for storing chat history
    if 'messages_memory' not in st.session_state:
        st.session_state.messages_memory = []

    # Display chat history
    for role, message in st.session_state.messages_memory:
        with st.expander(role.capitalize()):
            st.write(message)

    # Input box for user question
    user_input = st.text_area(':sneezing_face:请将你的疑问放进去(问问吧):', '')

    # Button to trigger response
    if st.button(':man-juggling:获取答案！'):
        if user_input:
            # Add user message to chat history
            st.session_state.messages_memory.append(("👩‍🔧投喂人", user_input))

            # Get response from ZhipuAI
            response = zhipu_chat_commemorate(user_input, temperatureChoicePage,
                                              promptAru=ChoosingPrompt)

            # Add ZhipuAI response to chat history
            st.session_state.messages_memory.append(("😼心脉AI", response))

        else:
            st.write('心脉AI没有检测到鱼干🐠')

#Page 5,langgraph开发
def langgraphRAG ():
    st.title("🐯心脉AI-Agent")
    st.subheader("基于 :grey[LangGraph]")

    if 'messages_rag' not in st.session_state:
        st.session_state.messages_rag = []

    # Display chat history
    for role, message in st.session_state.messages_rag:
        with st.expander(role.capitalize()):
            st.write(message)

    # Input box for user question
    user_input = st.text_area(':grinning_face_with_one_large_and_one_small_eye:请将你的小鱼干放进去(问问吧):', '')

    # Button to trigger response
    if st.button('👩‍🚀获取答案！'):
        if user_input:
            # Add user message to chat history
            st.session_state.messages_rag.append(("🥰投喂人", user_input))

            # Get response from ZhipuAI
            response = chatBot(user_input)

            # Add ZhipuAI response to chat history
            st.session_state.messages_rag.append(("😾心脉AI", response["messages"][0]))

        else:
            st.write('心脉AI没有检测到鱼干🐠')






# pageThree,后续指南
def page_using_mind():
    st.title(":red[大模型]使用指南")
    st.title(":orange[心脉AI]:cat:大模型使用指南")
    st.divider()

    st.markdown(
        """
        <div style='height: 20px; background: linear-gradient(to right, #ff7e5f, #feb47b);'></div>
        """,
        unsafe_allow_html=True)

    st.divider()
    st.subheader("第一项----:red[RAG]使用指南:sparkles:------:")
    st.text("感谢您使用心脉AI大模型，我们团队在此向您表示诚挚的感谢！")
    st.text("正如您所见,我们的模型的重点在于使用了RAG技术和streamlit前端，并使用Python进行整合。")
    st.text("以达到满意的效果。")
    st.text("对于普通用户来说,怎么使用呢？")
    st.text("鉴于streamlit本身具有的延时性,我们可以通过一下这三步,很简单:")
    st.text("传入问题、click、清空搜索栏 click again(这一步很重要 ！)")
    st.text("对于开发者的'专业版’而言,模型交给他们很多的函数接口,他们可以通过外部'遥控器'")
    st.text("直接调整大模型。胖橘的潜能可以在很大程度上被激发。")
    st.subheader("第二项----:orange[技术]实现流程:octopus:------:")
    st.text("一、大模型选择：在面临大模型的选择时，我们坚持使用国货:智谱Ai。它支持相对复杂的自然语言指令,")
    st.text("   广泛应用于智能客服等领域本次模型调用接口,我们胖橘使用了GLM-4作为后端底座。")
    st.text("二、团队在使用langchain进行各种文本处理时，需要将文本数据转化为向量形式。")
    st.text("   在使用了Huggingface_embedding,Openai_embedding后,我们发现zhipu的开放平台将")
    st.text("   embedding进行了封装,并将这个嵌入式底座和知识库直接连接,用户可以通过直接将例如pdf,doc,txt等")
    st.text("   数据上载到平台,直接调用，一方面省去了使用langchain进行load,split,embed...的手段")
    st.text("   另一方面,官方的数据处理更加专业,可以让我们的模型更加有利于增强检索。")
    st.divider()
    st.subheader("第三项----:blue[Langchain]增强模型能力:frog:-----")
    st.text("在大模型的构建中,当我们接入大模型时,最为重要的一点在于记忆性和模型加持")
    st.text("所以,在大模型中，我们使用langchain封装ZhipuAI接口,发送请求并接受响应")
    st.text("并且,我们使用langchain的ConversationBufferMemory类进行对话状态的存储,允许系统存储")
    st.text("之前的对话内容。并使用RunnableWithMessageHistory进行记忆功能。")
    st.text("使用invoke方法,用来传递给对话系统以进行交互。")
    st.divider()
    st.subheader("第四项----:green[streamlit]的前端实现:honeybee:-----")
    st.text("streamlit是一个强大的前端创建工具,我们团队使用其进行页面前端的创建。")
    st.text("我们使用其进行了实时对话,和Web部署等相对丰富的页面设计。")
    st.text("我们还使用了其第三方库进行动画的创作,交互性强。")
    st.divider()
    st.subheader("第五项----:grey[文档]结语:necktie:")
    st.text("我们团队在盛夏相识,在一次探索课中诞生了这个想法。这个作品说实话算不上工程,")
    st.text("最多是一次勇敢的尝试,和目前市面上的大模型相比相见形拙,并且,这个大模型知识库主要用于")
    st.text("高血压疾病的RAG检索,可以说,它是一个'专科'模型。")
    st.text("一个专门用途的大模型不叫”大模型“,同样,一个残缺的作品不是终点,而是起点。")

    st.text("以下是所有参与人员名单:")
    st.text("<<<开发团队(BUPT[北京邮电大学]):")
    st.text("---leader,programmer:  Mr.Chlorine")
    st.text("------team members,collector:  Ms.Chloe,Mr.Langongyao")

    st.text("在这里一并谢谢大家！")

    st.text("没有大家的热心打磨,齐心协力,就没有这个作品的实现与完善,再到上载。")


    st.text("就这样吧,如果您有更好的想法和建议,欢迎您和我们交流,再次感谢您的支持,后会有期！")
    st.text("---------------------------------------------------------笔者邮箱:2532807089@qq.com")
    st.divider()
    st.text("我们已经将代码开源到Github中,后续我们也会提供相关视频在B站,我们相信你也可以实现自己的大模型！")



# Navigation


# 面向开发者de代码模块,分为俩部分,第一个为Cookbook模块主要用于文档建设,第二个为
# 控制台函数,主要用于程序代码的修改。
# 同时,Cookbook将会对关键代码进行注解::
def developer_Cookbook():
    st.title("面向开发者的Cookbook")
    st.subheader("", divider='rainbow')
    st.subheader(":orange[心脉AI]使用多个函数进行修改,")
    st.subheader("更好的为您进行模型调优。")
    st.subheader("", divider='gray')
    st.subheader("⏰'拉杆'温度调节")
    st.text("在'医师'页面中,我们开发了temperature组件为您进行")
    st.text("大模型的温度改变,您可以直接拖动类似于如下的'拉杆',个性")
    st.text("化模型的输出,调整随机性,让胖橘更好的为您服务:")

    # 获取滑块的
    # 生成一个温度列表，基于滑块值
    temp = st.slider("模型温度", 0.0, 2.0, 0.1)
    slider_choice(temp)
    # 只显示滑块选择的温度值
    st.write("心脉AI的随机温度为 " + str(temp) + "℃.")

    st.subheader("", divider='blue')

    # 第二方面,使用token进行数据修改
    st.subheader("📇Tokens个性选择")
    st.text("在这里,我们使用了一个文本框来使得开发者能够自由选取Tokens,")
    st.text("来创建他们自己想要的回答长度。")
    st.number_input("输入你的Tokens")

    # 第三方面,使用prompt进行数据的更改
    st.subheader("", divider='red')
    st.subheader("📍Prompt定向查找")
    st.text("我们使用langchain的RunnableWithMessageHistory作为")
    st.text("记忆模块,这样就可以达到记忆存储的结果。")
    st.text("如同Java的Runnable接口一样,这个类的内部封装了一个Prompt。")
    st.text("我们在这里创建了一个接口,通过调用函数接口对这个类的内部直接进行修改,使得")
    st.text("开发者可以直接面向问题查询,极大的保证了问题的准确性。")

    st.text_input("你的Prompt问题是什么")
    pass


# """
# 控制台函数
# """


# 记忆模型调参函数
def sliderTem():
    tem = st.slider("选择temperature进行模型内部调参", 0.0, 2.0, 0.1)
    return tem


# Cookbook调参函数
# First.使用slider进行温度随机性调节
def add_message_decorator(func):
    def wrapper(temp):
        func(temp)

        if temp == 2.0:
            st.write("温度过高😂,心脉AI会发烧")
        elif temp == 1.0:
            st.write("温度最好了🥳,建议使用的temperature")
        elif temp == 0.0:
            st.write("温度过低⛄️,心脉AI冬眠。一觉醒来")
            st.write("最好的朋友在身边,最爱的人在对面")
            st.write("愿我们都能成为更好的自己。💪")
        else:
            st.write("继续拉动'滑杆'为心脉AI调参吧🙌")

    return wrapper


@add_message_decorator
def slider_choice(temp):
    if 0 < temp < 0.5:
        st.write("现在的心脉AI温度不高")
    elif 0.5 <= temp < 1.0:
        st.write("心脉AI有点来感觉了呃")
    elif temp == 1.0:
        st.write("心脉AI正在兴头上，请放心使用")
    elif 1.0 < temp < 2.0:
        st.write("心脉AI正在天马行空")


# Sec.Token修改
def TokenChoice():
    catFeed = st.text_input("请先修改心脉AI😸的日常猫粮Tokens使用额度再询问问题！",
                            max_chars=55)
    if catFeed == 0:
        return 500
    st.button("😿提交你的心脉AI额度!😼")
    return catFeed


# Thr.使用不同提示词进行记忆检索
def prompt_memorize():
    Looking = st.text_input("你对心脉AI的最新期待(Prompt)")
    # 这里直接继承到Cookbook的方法
    return Looking


# FOur.哈希密码创建
def hash_password(password):  # 生成一个随机salt:
    salt = os.urandom(16)  # 生成16字节的随机盐值
    # 将密码和盐值合并
    salted_password = salt + password.encode('utf-8')
    # 使用SHA256哈希算法
    hash_obj = hashlib.sha256(salted_password)
    # 获取16字节的哈希值
    hash_value = hash_obj.digest()
    # 将盐值和哈希值拼接在一起
    return salt + hash_value


# 定义一个函数来验证密码
def verify_password(stored_value, password):
    salt = stored_value[:16]  # 提取盐值
    hash_value = stored_value[16:]  # 提取哈希值
    salted_password = salt + password.encode('utf-8')
    hash_obj = hashlib.sha256(salted_password)
    return hash_obj.digest() == hash_value


def main():
    background_css = """
    <style>
    body {
        background-image: url('E:\机器学习\探索课大作业\\final\cat.jpg');
        background-repeat: no-repeat;
        background-size: cover;
    }
    </style>
    """
    if "role" not in st.session_state:
        st.session_state.role = None

    # 在Streamlit应用中插入CSS样式
    st.markdown(background_css, unsafe_allow_html=True)

    ROLES = ["🕵️‍♂️患者", "👨‍🎓医师", "👨‍🦯游客"]  # 角色选项

    def login():
        nameBase = ['Charlie', 'Chloe', 'Chlorine', 'Langongyao', 'Chelsea',
                    'Teacher', 'Dick', 'May', 'Dingding', 'Harris', 'Rugosa',
                    'LooseAnus','Moran']

        # 定义每天的话语
        daily_messages = {
            0: "星期一：我愿永远活在心脉AI之中💝",
            1: "星期二：今天也要努力搬砖鸭👨‍🎨",
            2: "星期三：不要忘了给心脉AI投喂小鱼干呀🐟",
            3: "星期四：啊啊~心脉AI我的家🏫",
            4: "星期五：为什么大家都爱用心脉AI👾",
            5: "星期六：坚强的、努力的、向上的🔮",
            6: "星期天：心脉AI的故事📌",
               }

        # 获取当前的星期几
        today = datetime.datetime.today().weekday()
        # 获取对应的话语
        message = daily_messages.get(today, "今天没有特别的话语。")
        # 使消息居右显示的CSS样式
        message_css = """
        <style>
        .right-message {
            position: absolute;
            top:1 px;
            right: 20px;
            font-size: 20px;
            color: purple; /* 可以根据背景颜色调整文本颜色 */
        }
        </style>
        """
        st.markdown(message_css , unsafe_allow_html=True)
        st.markdown(f'<div class="right-message">{message}</div>', unsafe_allow_html=True)

        st.header("🦄:orange[心脉AI]大模型")
        st.subheader("一款面向高血压疾病的“_:blue[专科]_”大模型!", divider='rainbow')

        st.caption("🤔请登录以继续:")

        role = st.selectbox(":blue[请选择您的角色]🤗 ", ROLES)
        if role == "👨‍🎓医师":
            extra_info = st.text_input("请输入您的姓名(直接输入'May'即可)")

            for name in nameBase:
                if extra_info == name:
                    hashed_password = hash_password(extra_info)
                    st.write("你的验证码为:" + hashed_password.hex()[:5])
                    dictator = st.text_input("请将验证码交给胖橘查阅~")
                    if dictator:
                        if verify_password(hashed_password, extra_info):
                            st.write("欢迎开发者" + extra_info + "回家！胖橘将会温暖你的心哟！")

                            if st.button("加入心脉AI😀"):
                                st.session_state.role = role

                                st.experimental_rerun()
                        else:
                            st.write("验证码不正确,请重试")
        elif role == "🕵️‍♂️患者":
            if st.button("加入心脉AI😀"):
                st.session_state.role = role
                st.experimental_rerun()
        elif role == "👨‍🦯游客":
            st.write("请选择用户来进行登录")

        st.divider()
        st.text("'患者'面向普通用户,'医师'面向医师和开发者。")
        st.text("其中12位用户拥有查阅Cookbook并可以通过函数接口直接微调大模型的权限")
        st.text("请开发者选择'医师'并在弹出的消息框中输入姓名,为了安全性,系统自动会给出")
        st.text("一组实时验证码,开发者将其输入即可进入。")

        st_lottie("https://lottie.host/bbbb09a5-4099-4b1b-98b6-625c121b7769/QxDcoA4tIw.json")

    def logout():
        st.session_state.role = None
        st.experimental_rerun()

    st.sidebar.title("🦹‍♂️:red[心脉AI导航]")
    if st.session_state.role == "🕵️‍♂️患者":
        page = st.sidebar.radio("📊请你选择使用的大模型",
                                ['心脉AI RAG 智能检索大模型', '心脉AI记忆检索大模型' ,
                                 '大模型使用指南'
                                 ])
    else:
        page = st.sidebar.radio("📊请你选择使用的大模型",
                                ['心脉AI RAG 智能检索大模型', '心脉AI记忆检索大模型','心脉AIAgent' ,
                                 '大模型使用指南', '面向开发者的Cookbook'])

    # 如果用户未登录，显示登录界面
    #😺😸😹😻😼😽🙀😿😾🐱
    if st.session_state.role is None:
        login()
    else:
        # 否则显示主页面和侧边栏导航
        st.sidebar.button("离开小胖橘", on_click=logout)

        # 根据用户角色显示对应页面
        if st.session_state.role == "🕵️‍♂️患者":

            if page == '心脉AI RAG 智能检索大模型':
                tavily_election = "Closed"
                about_token = 5000
                page_zhipuai_rag(about_token, tavily_election)

            elif page == '心脉AI记忆检索大模型':

                page_zhipuai_memory(temperatureChoicePage=0.5, ChoosingPrompt="高血压")

            #elif page == '胖橘Agent':
            #   langgraphRAG()

            elif page == '大模型使用指南':
                page_using_mind()
        elif st.session_state.role == "👨‍🎓医师":
            if page == '心脉AI RAG 智能检索大模型':
                tavily_election = "Opened"
                #在这块添加一个判断额度
                #@author:2024.7.15 21:58修改
                #主要修改为判断语句进行BUG修复
                about_token = TokenChoice()
                if about_token == 0:
                    page_zhipuai_rag(500, tavily_election)
                else:
                    page_zhipuai_rag(about_token, tavily_election)

            elif page == '心脉AI记忆检索大模型':
                tem = sliderTem()
                this_prompt_memorize = prompt_memorize()
                page_zhipuai_memory(temperatureChoicePage=tem, ChoosingPrompt=this_prompt_memorize)

            elif page == '心脉AIAgent':
                langgraphRAG()

            elif page == '大模型使用指南':
                page_using_mind()
            elif page == '面向开发者的Cookbook':
                developer_Cookbook()

        else:
            st.write("您没有权限访问此页面。")

    st.sidebar.markdown('<style>div.Widget.row-widget.stRadio > div{background-color: #add8e6}</style>',
                        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
