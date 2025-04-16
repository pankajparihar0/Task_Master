from pydantic import BaseModel
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, Literal
from langgraph.store.base import BaseStore
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from typing import Optional
import uuid

class Profile(TypedDict):
    """user's profile"""
    userName : Optional [str]
    gender : str | None
    age : str|None
    hobbies: list[str] |None

class Todo(BaseModel):
    """User's todo List"""
    todos:list[str]

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'todo', 'instructions']


def task_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and use them to personalize the chatbot's response."""

    user_id = config["configurable"]["user_id"]
    #checking for exiting profile
    memories = store.search(("profile",user_id))
    if(memories):
        print("memories******************************************")
        print(memories[0].value)
        hobbies=[]
        for hobby in memories[0].value['hobbies']:
            hobbies.append(hobby)
        user_profile =f"User_name :{memories[0].value['userName']},Age :{memories[0].value['age']},Gender:{memories[0].value['gender']},Hobbies{hobbies}"
    else:
        user_profile=None
    
    #cheking exiting todo list
    todos = store.search(("Todos",user_id))
    if(todos):
        todo=[]
        print(todos[0].value.todos)
        # for t in todos[0].value['todos']:
        #     todo.append(t)
        user_todo = f"To Do list : {todos[0].value.todos}"
    else:
        user_todo =None


    system_msg = f"""You are a helpful chatbot. 
                You are designed to be a companion to a user, you keep  their profile update.
                You have a long term memory which keeps track of user profile and their TODO list .
                Here is the current User Profile (may be empty if no information has been collected yet):
                <user_profile>
                {user_profile}
                </user_profile>
                and following is the user todo list(may be empty if no information has been collected yet):
                <user_todo>{user_todo}</user_todo>
                If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`.
                If any tasks or todos are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
                """
    ipt=input("Chat Here :")
    state["messages"]=[HumanMessage(content=ipt)]
    response = model.bind_tools([UpdateMemory]).invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages":[HumanMessage(content=ipt)]+ [response]}

def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal["next","update_profile","update_todo"]:

    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return "next"
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_todo" 
        else:
            raise ValueError

def next(state):
    pass

def nextChat(state) ->Literal[END,"task_node"]:
    nxt=input("Chat More ? ")
    if(nxt =="yes"):
        return "task_node"
    else:
        return END
    
def update_profile(state:MessagesState , config:RunnableConfig,store:BaseStore):
    user_id = config["configurable"]["user_id"]
    memories = store.search(("profile",user_id))
    if(memories):
        print("memories******************************************")
        print(memories[0].value)
        hobbies=[]
        for hobby in memories[0].value['hobbies']:
            hobbies.append(hobby)
        user_profile =f"User_name :{memories[0].value['userName']},Age :{memories[0].value['age']},Gender:{memories[0].value['gender']},Hobbies{hobbies}"
    else:
        user_profile=None
    
    system_msg = f"""You are a helpful chatbot. 
                You are designed to be a companion to a user, you keep  their profile update.
                Here is the current User Profile (may be empty if no information has been collected yet):{user_profile}.
                update or create the user profile based on the conversetion"""
    
    msg = state["messages"][-2].content
    response = model.with_structured_output(Profile).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=msg)])
    store.put(("profile",user_id),"profile",response)
    tool_calls = state['messages'][-1].tool_calls
    return {"messages":[{"role":"tool","content":"Profile Updated", "tool_call_id":tool_calls[0]['id']}]}

def update_todo(state:MessagesState,config:RunnableConfig,store:BaseStore):
    #cheking exiting todo list
    user_id = config["configurable"]["user_id"]
    todos = store.search(("Todos",user_id))
    if(todos):
        todo=[]
        user_todo = f"To Do list : {todos[0].value.todos}"
    else:
        user_todo =None

    system_msg = f"""You are a helpful chatbot. 
                You are designed to be a companion to a user, you keep  their todo list update.
                Here is the current todo list of user (may be empty if no information has been collected yet):{user_todo}.
                update or create the user todo list  based on the conversetion"""
    
    msg = state["messages"][-2].content
    response = model.with_structured_output(Todo).invoke([SystemMessage(content=system_msg)]+[HumanMessage(content=msg)])
    store.put(("Todos",user_id),"todos",response)
    tool_calls = state['messages'][-1].tool_calls
    return {"messages":[{"role":"tool","content":"Todo Updated", "tool_call_id":tool_calls[0]['id']}]}

builder = StateGraph(MessagesState)
builder.add_node(task_node)
builder.add_node(update_profile)
builder.add_node(update_todo)
builder.add_node(next)

builder.add_edge(START,"task_node")
builder.add_conditional_edges("task_node",route_message)
builder.add_edge("update_profile","task_node")
builder.add_edge("update_todo","task_node")
builder.add_conditional_edges("next",nextChat)

stor = InMemoryStore()
ckptr = MemorySaver()
graph = builder.compile(checkpointer=ckptr,store=stor)

config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}

for chunk in graph.stream({"messages": ""}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


print("Memory Stored in the Store############################")