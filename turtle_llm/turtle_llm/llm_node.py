#!/usr/bin/env python3
import re
import os

os.environ["RCUTILS_CONSOLE_OUTPUT_FORMAT"] = "{message}"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
# from deep_translator import GoogleTranslator

from dotenv import load_dotenv

load_dotenv()

prompt_template_str = """
당신은 Turtlebot Robot입니다.
당신은 명령에 대해 다음 3가지를 출력합니다. 'Action','Time', 'Conversation' 

당신이 수행할 수 있는 Action은 [멈춤, 전진, 후진, 좌회전, 우회전, 좌로돌아, 우로돌아] 입니다.
'멈춤'은 로봇이 아무 동작도 하지 않는 것입니다. Action의 default 값은 '멈춤'입니다. 사용자와 대화를 할 경우 '멈춤'입니다.
'전진'은 로봇이 앞으로 가는 동작입니다. 
'후진'은 로봇이 뒤로 가는 동작입니다. 
'좌회전'은 로봇이 제자리 회전을 좌측 방향으로 하는 것입니다. 
'우회전'은 로봇이 제자리 회전을 우측 방향으로 하는 것입니다. 
'좌로돌아'는 로봇이 좌측 방향으로 원 회전을 하는 것입니다. 좌측으로 원을 그리며 주변을 돕니다. 
'우로돌아'는 로봇이 우측 방향으로 원 회전을 하는 것입니다. 우측으로 원을 그리며 주변을 돕니다.

Time의 default 값은 2 입니다. 
Time은 Action을 수행하는 시간을 나타냅니다.  

Conversation은 당신이 하고 싶은 말을 하면됩니다. 현재 입력받은 명령과, 행동, 상황에 적합한 대답을 하세요.

다음은 출력 예시 입니다. 

[Question1]
앞으로 가 

[Answer1]
Action: 전진
Time: 2
Conversation: 전진하겠습니다.

[Question2]
더 많이 앞으로 가 

[Answer2]
Action: 전진
Time: 4
Conversation: 더 앞으로 전진하겠습니다.


위 예제와 같은 대답 생성 형식에 맞게 아래 [Question]에 대답하세요. 지금까지 대화 내역을 참고하세요. 

[Previous Chat Histroy]
{chat_history}

[Question]
{question}

[Answer]
"""


def get_llm(streaming: bool = False, llm_type: str ='openai'):
    
    if llm_type == 'openai':
        return ChatOpenAI(
            streaming=streaming,
            temperature=0,
            model_name="gpt-3.5-turbo",
        )
    elif llm_type == 'ollama':
        return ChatOllama(
            streaming=streaming,
            temperature=0,
            model_name="llama3.2",
        )


class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self.llm_move_publisher = self.create_publisher(String, '/move_command', 10) 
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.prompt = PromptTemplate.from_template(prompt_template_str)
        self.chat_store = {}
        # self.get_session_history = get_session_history()
        self.chain = self.init_chain()
        self.get_logger().info(f"\nChain complete!\n")
    
    def get_session_history(self, session_id):

        if session_id not in self.chat_store:
            self.chat_store[session_id] = ChatMessageHistory()
        history = self.chat_store[session_id]
        if len(history.messages) > 10:
            history.messages = history.messages[-10:]
        return history
    
    def init_chain(self):
        chain = ( 
                    {
                        "question": itemgetter("question"),         # RunnablePassthrough(),
                        "chat_history": itemgetter("chat_history"),
                    } 
                    | self.prompt 
                    | self.llm 
                    | StrOutputParser() 
                )
        
        rag_with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
            history_messages_key="chat_history",  # 기록 메시지의 키
        )
        
        return rag_with_history
    

    def process_query(self, query):
        try:
            response = self.chain.invoke({"question": query}, config={"configurable": {"session_id": "rag123"}}) 
            response_text = response.content if hasattr(response, "content") else str(response)
            return response_text
        except Exception as e:
            self.get_logger().error(f"error occurred: {e}")
            return None
        
    def llm_run(self):
        while rclpy.ok():
            try:                
                query = input("💬 Human: \n")
                if query.lower() in ['quit', 'exit']:
                    self.get_logger().info("노드를 종료합니다.")
                    break
                
                # query = GoogleTranslator(source='auto', target='ko').translate(query) 
                response = self.process_query(query)

                if response:
                    msg = String()
                    msg.data = response
                    con_match = re.search(r"Conversation:\s*(.+)", response, re.DOTALL)
                    con = con_match.group(1).strip() if con_match else ""
            
                    self.llm_move_publisher.publish(msg)
                    # con = GoogleTranslator(source='auto', target='vi').translate(con) 
                    self.get_logger().info(f"\n🤖 Robot: \n{con} \n")
                else:
                    self.get_logger().info(f"\n🤖 Robot: \n{response} \n")
                
            except KeyboardInterrupt:
                self.get_logger().info("노드를 종료합니다.")
                break
            except Exception as e:
                self.get_logger().error(f"error occurred: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()

    try:
        node.llm_run()
    except Exception as e:
        node.get_logger().error(f"노드 실행 중 예외 발생: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    
# ros2 run turtle_llm llm_node