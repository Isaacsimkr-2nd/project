#!/usr/bin/env python3
import os
os.environ["RCUTILS_CONSOLE_OUTPUT_FORMAT"] = "{message}"
import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from langchain_openai import ChatOpenAI
from turtle_agent.Agents.executor import ActionExecutor
from turtle_agent.Agents.planner import Planner
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class AgentNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self.llm_move_publisher = self.create_publisher(String, '/move_command', 10) 
        
        self.session_id = str(random.randrange(1, 100))
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.verbose = False

        self.planner = Planner(llm=self.llm, session_id=self.session_id)
        self.action_executor = ActionExecutor(llm=self.llm, session_id=self.session_id, verbose=self.verbose )
        
        self.get_logger().info("\n✅ Chain complete!\n")

    def agent_run(self):
        while rclpy.ok():
            try:                
                query = input("🗣️ Human: \n")
                if query.lower() in ['quit', 'exit']:
                    self.get_logger().info("🔴 노드를 종료합니다.")
                    break

                final_plan = self.planner.plan(query)[0]
                goal = final_plan['GOAL']
                plan = final_plan['PLAN']
                print_plan = " - ".join([f"[{item}]" for item in plan])
                self.get_logger().info(f"\n🎯 GOAL >> {goal}")
                self.get_logger().info(f"\n📌 PLAN >> {print_plan} \n")
                self.get_logger().info(f'>>>>'*10)
                
                # ActionExecutor를 사용하여 실행
                result = None
                for step in plan:
                    self.get_logger().info(f"\nSTEP: {step}\n")
                
                    result = self.action_executor.invoke(query, goal, plan, step, result)
                    self.get_logger().info(f"\n🤖 System >> \n{result}\n")
                    self.get_logger().info(f'>>>>'*10)

                
            except KeyboardInterrupt:
                self.get_logger().info("🔴 노드를 종료합니다.")
                break
            except Exception as e:
                self.get_logger().error(f"❌ error occurred: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()

    try:
        node.agent_run()
    except Exception as e:
        node.get_logger().error(f"❌ 노드 실행 중 예외 발생: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
