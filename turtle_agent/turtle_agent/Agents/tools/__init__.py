
from typing import Literal, List, Optional
from langchain.agents import Tool



class ToolBox:
    def __init__(
        self
    ):
        self.__tools: list = []

        # Add the default tools
        from . import tools
        self.__iterative_add(tools)
  
    def get_tools(self) -> List[Tool]:
        return self.__tools

    def __add_tool(self, tool):
        if hasattr(tool, "name") and hasattr(tool, "func"):
            self.__tools.append(tool)

    def __iterative_add(self, package):
        """
        Iterate through a package and add each @tool to the tools list.

        :param package: The package to iterate through.
        :param blacklist: A parameter used by some tools to filter out certain results.
        """
        for tool_name in dir(package):
            if not tool_name.startswith("_"):
                t = getattr(package, tool_name)
                self.__add_tool(t)

    def add_packages(self, tool_packages: List, blacklist: Optional[List[str]] = None):
        """
        Add a list of tools to the Tools object by iterating through each package.

        :param tool_packages: A list of tool packages to add to the Tools object.
        """
        for pkg in tool_packages:
            self.__iterative_add(pkg, blacklist=blacklist)

    def add_tools(self, tools: list):
        """
        Add a single tool to the Tools object.

        :param tools: A list of tools to add
        """
        for tool in tools:
            self.__add_tool(tool)
