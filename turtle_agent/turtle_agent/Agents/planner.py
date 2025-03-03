from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


import json
import argparse
import random
from dotenv import load_dotenv

load_dotenv()

from .prompts.prompts import planner_prompts

def custom_json_parser(response):
    json_string = (
        response.content.strip()
        .removeprefix("```json\n")
        .removesuffix("\n```")
        .strip()
    )
    json_string = f"[{json_string}]"
    return json.loads(json_string)


class Planner:
    def __init__(self, llm, session_id: str = "default"):
        self.session_id = session_id
        self.last_plan = None  # 마지막 계획 결과를 저장할 변수
        self.__llm = llm
        self.__prompts = self._get_prompts() 
        self.__chain = self._get_chain()
    
    def _get_prompts(self) -> ChatPromptTemplate:
        """Create a chat prompt template from the system prompts and robot-specific prompts."""
        # Start with default system prompts
        prompts = planner_prompts

        template = ChatPromptTemplate.from_messages(
            prompts
            + [
                ("placeholder", "{chat_history}"),
                ("human", "{command}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        return template
    
    def _get_chain(self):
        # 체인 구성: RunnablePassthrough → 프롬프트 → LLM → JSON 파서
        chain = (
            {"command": RunnablePassthrough()}
            | self.__prompts
            | self.__llm
            | custom_json_parser
        )

        # 대화 기록 생성 및 체인에 연결
        chat_history = ChatMessageHistory()
        agent_with_chat_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: chat_history,
            input_messages_key="command",
            history_messages_key="chat_history",
        )
        return agent_with_chat_history
    
    
    def plan(self, command: str) -> str:
        # .invoke()를 사용해 전체 결과를 받아옵니다.
        self.last_plan = self.__chain.invoke(
            {"command": command},
            config={"configurable": {"session_id": self.session_id}}
        )
        return self.last_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planner command-line options")
    args = parser.parse_args()

    session_id = str(random.randrange(1, 100))
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    
    planner = Planner(llm=llm, session_id=session_id)

    print("Welcome to the planner! To exit, type 'exit' or 'quit'.\n")

    while True:
        command = input("🗣️ Command: \n")
        if command.lower() in ["exit", "quit"]:
            print("Exiting the planner")
            break
        
        final_plan = planner.plan(command)
        print("Final plan:", final_plan)
        print()
