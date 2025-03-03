from langchain.tools import tool
from typing import Any, AsyncIterable, List, Dict, Literal, Optional, Union
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks import get_openai_callback  
from langchain_core.messages import AIMessage, HumanMessage  
from langchain_teddynote.messages import AgentCallbacks, AgentStreamParser
from langchain.prompts import MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

import random
import argparse
from dotenv import load_dotenv

load_dotenv()

import rclpy
from rclpy.node import Node

from .tools import ToolBox 
from .prompts.prompts import RobotSystemPrompts, executor_prompts



# === 챗봇 에이전트 클래스 정의 ===
class ActionExecutor:
    def __init__(
        self, 
        llm,
        tools: Optional[list] = None,
        tool_packages: Optional[list] = None,
        prompts: Optional[RobotSystemPrompts] = None,
        session_id: str = "default", 
        accumulate_chat_history: bool = True,
        verbose: bool = True,
    ):
        self.__chat_history = []  
        self.__memory_key = "chat_history"
        self.__scratchpad = "agent_scratchpad"
        self.__accumulate_chat_history = accumulate_chat_history
        self.__session_id = session_id
        self.__verbose = verbose
        self.last_answer = None                         # 마지막 응답을 저장할 변수
        
        self.__llm = llm
        self.__prompts = self._get_prompts(prompts)     # 프롬프트 템플릿 생성
        self.__toolbox = self._get_tools(packages=tool_packages, tools=tools)
        self.__tools = self.__toolbox.get_tools()
        self.__llm_with_tools = self.__llm.bind_tools(self.__tools)     # LLM과 도구 결합
        self.__agent = self._get_agent()
        self.__executor = self._get_executor()
        
    
    def _get_tools(
        self,
        packages: Optional[list],
        tools: Optional[list],
    ) -> ToolBox:
        """Create a ROSA tools object with the specified ROS version, tools, packages, and blacklist."""
        rosa_tools = ToolBox()
        if tools:
            rosa_tools.add_tools(tools)
        if packages:
            rosa_tools.add_packages(packages)
        return rosa_tools
    
    def _get_prompts(
        self, robot_prompts: Optional[RobotSystemPrompts] = None
    ) -> ChatPromptTemplate:
        """Create a chat prompt template from the system prompts and robot-specific prompts."""
        # Start with default system prompts
        prompts = executor_prompts

        # Add robot-specific prompts if provided
        if robot_prompts:
            prompts.append(robot_prompts.as_message())

        template = ChatPromptTemplate.from_messages(
            prompts
            + [
                MessagesPlaceholder(variable_name=self.__memory_key),
                ("human", "{pre_result}"),
                MessagesPlaceholder(variable_name=self.__scratchpad),
            ]
        )
        return template
    
    def _get_agent(self):
        """Create and return an agent for processing user inputs and generating responses."""
        # agent = create_tool_calling_agent(self.__llm, self.__tools, self.__prompts)

        agent = (
            {
                "instruction": lambda x: x["instruction"],
                "goal": lambda x: x["goal"],
                "plan": lambda x: x["plan"],
                "step": lambda x: x["step"],
                "pre_result": lambda x: x["pre_result"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | self.__prompts
            | self.__llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        
        return agent
    

    def _get_executor(self) -> AgentExecutor:
        """Create and return an executor for processing user inputs and generating responses."""
        executor = AgentExecutor(
            agent=self.__agent,
            tools=self.__tools,
            verbose=self.__verbose,
            max_iterations=10,
            max_execution_time=10,
            handle_parsing_errors=True,
        )

        return executor
    
    
    def invoke(self, instruction: str, goal: str, plan: str, step: str, pre_result: str) -> str:

        try:
            with get_openai_callback() as cb:
                # result = self.__executor.invoke(
                #     {"input": query, "chat_history": self.__chat_history}
                # )
                result = self.__executor.invoke(
                    {"instruction": instruction, "goal": goal, "plan": plan, "step": step, "pre_result": pre_result, "chat_history": self.__chat_history}
                )
        except Exception as e:
            return f"An error occurred: {str(e)}"

        self._record_chat_history(instruction, result["output"])
        return result["output"]
    
    

    def _record_chat_history(self, query: str, response: str):
        """Record the chat history if accumulation is enabled."""
        if self.__accumulate_chat_history:
            self.__chat_history.extend(
                [HumanMessage(content=query), AIMessage(content=response)]
            )
    
    def chat(self, instruction: str, goal: str, plan: str, step: str, pre_result: str) -> str:
        self.last_answer = None  # 이전 결과 초기화
        
        response_stream = self.__executor.invoke(
                {"instruction": instruction, "goal": goal, "plan": plan, "step": step, "pre_result": pre_result},
                config={"configurable": {"session_id": self.__session_id, "stream": True}},
            )
            
        return response_stream


# === 메인 실행부 ===
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ActionExecutor command-line options")
    parser.add_argument("-v", "--verbose", type=str, default='True', help="Enable verbose mode (default: False)")
    args = parser.parse_args()

    verbose = True if args.verbose == 'True' else False
    
    chat_ID = str(random.randrange(1,100)) # input("ID를 입력해주세요: ")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    
    chatbot = ActionExecutor(llm=llm, session_id=chat_ID, verbose=verbose)
    print("Welcome to the chatbot! To exit, type 'exit', or 'quit'.\n")

    while True:
        # 새 텍스트 입력 전에 이전에 공유했던 카메라 피드 창이 열려있다면 닫습니다.
        
        instruction = input("🗣️ instruction: \n")
        goal = input("🗣️ goal: \n")
        plan = input("🗣️ plan: \n")
        step = input("🗣️ step: \n")
        
        pre_result = input("🗣️ pre_result: \n")
        
        
        # stream 옵션에 따라 호출
        final_answer = chatbot.invoke(instruction, goal, plan, step, pre_result)
        
        