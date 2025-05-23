"""
AI Services Module
Handles LangChain SQL agent configuration and AI service interactions.
"""

import os
import logging
from typing import Optional, List

# LangChain imports
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.callbacks.base import BaseCallbackHandler

# SQLAlchemy imports
from sqlalchemy import create_engine, MetaData, Table

try:
    from config import config
    from exceptions import AIServiceError
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.dirname(__file__))
    from config import config
    from exceptions import AIServiceError


class SQLQueryHandler(BaseCallbackHandler):
    """Custom callback handler to capture SQL queries during agent execution"""
    
    def __init__(self):
        self.sql_queries = []
        self.sql_results = []
        
    def on_agent_action(self, action, **kwargs):
        """Capture SQL query when the tool being used is sql_db_query"""
        if action.tool in ["sql_db_query"]:
            self.sql_queries.append(action.tool_input)
            logging.debug("SQL Query captured: %s", action.tool_input)
    
    def on_tool_end(self, output, **kwargs):
        """Capture SQL query results"""
        self.sql_results.append(output)
        logging.debug("SQL Result captured")
    
    def clear(self):
        """Clear captured queries and results"""
        self.sql_queries.clear()
        self.sql_results.clear()
    
    def get_last_query(self) -> Optional[str]:
        """Get the last executed SQL query"""
        return self.sql_queries[-1] if self.sql_queries else None


class DatabaseService:
    """Database connection and SQLAlchemy setup service"""
    
    def __init__(self):
        self.connection_string = config.database.connection_string
        self.db_engine = None
        self.sql_database = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database engine and reflect tables"""
        try:
            # Create SQLAlchemy engine
            self.db_engine = create_engine(
                f'mssql+pyodbc:///?odbc_connect={self.connection_string}',
                pool_size=config.database.pool_size,
                pool_timeout=config.database.pool_timeout,
                echo=config.app.debug  # Log SQL queries in debug mode
            )
            
            # Reflect the Problems table with specific columns
            metadata = MetaData()
            self.problems_table = Table('Problems', metadata,
                autoload_with=self.db_engine, 
                schema='stage',
                include_columns=[
                    'ProblemID', 'Priority', 'State', 'Services', 'AssignmentGroup', 'Category',
                    'ResolutionCode', 'CreatedDateTime', 'ResolvedDateTime', 'ReassignmentCount',
                    'ReopenCount', 'RCAStartTime', 'RCAEndTime', 'MadeSLA', 'RelatedIncidents', 
                    'JiraID', 'Impact'
                ]
            )
            
            # Create SQLDatabase instance for LangChain
            self.sql_database = SQLDatabase(self.db_engine, metadata=metadata)
            
            logging.info("Database service initialized successfully")
            
        except Exception as e:
            logging.error("Failed to initialize database service: %s", e)
            raise AIServiceError(f"Database initialization failed: {e}") from e
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return result.fetchone()[0] == 1
        except Exception as e:
            logging.error("Database connection test failed: %s", e)
            return False
    
    def get_sql_database(self) -> SQLDatabase:
        """Get the LangChain SQLDatabase instance"""
        if not self.sql_database:
            raise AIServiceError("Database not initialized")
        return self.sql_database


class LLMService:
    """Azure OpenAI LLM service configuration"""
    
    def __init__(self):
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize Azure OpenAI LLM"""
        try:
            # Set environment variables for Azure OpenAI
            os.environ["AZURE_OPENAI_API_KEY"] = config.azure_openai.api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = config.azure_openai.endpoint
            os.environ["AZURE_OPENAI_API_VERSION"] = config.azure_openai.api_version
            os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = config.azure_openai.deployment_name
            
            self.llm = AzureChatOpenAI(
                azure_deployment=config.azure_openai.deployment_name,
                api_version=config.azure_openai.api_version,
                temperature=config.azure_openai.temperature,
                max_tokens=config.azure_openai.max_tokens,
                timeout=None,
                max_retries=config.azure_openai.max_retries,
            )
            
            logging.info("Azure OpenAI LLM initialized successfully")
            
        except Exception as e:
            logging.error("Failed to initialize LLM: %s", e)
            raise AIServiceError(f"LLM initialization failed: {e}") from e
    
    def get_llm(self) -> AzureChatOpenAI:
        """Get the configured LLM instance"""
        if not self.llm:
            raise AIServiceError("LLM not initialized")
        return self.llm


class SQLAgentService:
    """SQL Agent service with conversation management"""
    
    def __init__(self, database_service: DatabaseService, llm_service: LLMService):
        self.database_service = database_service
        self.llm_service = llm_service
        self.sql_agent = None
        self.sql_handler = SQLQueryHandler()
        self.memory = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the SQL agent with tools and memory"""
        try:
            # Get LLM and database
            llm = self.llm_service.get_llm()
            sql_database = self.database_service.get_sql_database()
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
            
            # Create SQL toolkit
            sql_toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm)
            
            # Create SQL agent
            self.sql_agent = create_sql_agent(
                llm=llm,
                toolkit=sql_toolkit,
                memory=self.memory,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=config.app.debug,
                handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
                allow_dangerous_requests=True,
                max_iterations=100,
                agent_executor_kwargs={
                    'handle_parsing_errors': "If successfully execute the plan then return summarize and end the plan. Otherwise, please call the API step by step.",
                    'callbacks': [self.sql_handler]
                }
            )
            
            logging.info("SQL Agent initialized successfully")
            
        except Exception as e:
            logging.error("Failed to initialize SQL agent: %s", e)
            raise AIServiceError(f"SQL Agent initialization failed: {e}") from e
    
    def query(self, question: str) -> dict:
        """Execute a query using the SQL agent"""
        if not self.sql_agent:
            raise AIServiceError("SQL Agent not initialized")
        
        try:
            # Clear previous query data
            self.sql_handler.clear()
            
            # Create the prompt
            formatted_prompt = self._create_prompt(question)
            
            # Execute the query
            response = self.sql_agent.invoke({"input": formatted_prompt})
            
            # Return structured response
            result = {
                "question": question,
                "response": response.get("output", ""),
                "sql_queries": self.sql_handler.sql_queries.copy(),
                "success": True
            }
            
            logging.info("SQL query executed successfully for question: %s", question[:100])
            return result
            
        except Exception as e:
            logging.error("SQL query execution failed: %s", e)
            return {
                "question": question,
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "sql_queries": self.sql_handler.sql_queries.copy(),
                "success": False,
                "error": str(e)
            }
    
    def _create_prompt(self, question: str) -> str:
        """Create formatted prompt for the SQL agent"""
        system_prompt = """
You are a helpful Agent designed to interact with a mssql database.
Given an input question, create a syntactically correct mssql query to run, then look at the results of the query and return the answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
Focus on the table - [stage].[Problems] The answers for All the user questions can be found in this table.

Below are the table details, the column details and detailed description:
-- Table and Column Details:
-- Problem Management Stage Table Name: [stage].[Problems]
-- Problem Management Column Names:
[ProblemID] -- Contains the unique identifier for each problem for example: PRB0044797, PRB0043935
[Priority] -- Contains the priority of the problem for example: 1-Critical, 2-High, 3-Medium, 4-Low
[State] -- Contains the state of the problem for example: New, In Progress, On Hold, Resolved, Closed
[Services] -- Contains the services for which problem is raised for example: GOE Hybris - PROD, Helpie Virtual Agent - PROD
[AssignmentGroup] -- Contains the group name to which problem is assigned for example: Problem Mgmt Local Operations Tooling
[Category] -- Contains the category of the problem for example: Incident, Service Request, Problem
[ResolutionCode] -- Contains the resolution code of the problem for example: Cancelled, Closed, Fix Applied, Risk Accepted
[CreatedDateTime] -- Contains the date and time when problem is created for example: 2022-01-01 00:00:00.000
[ReassignmentCount] -- Contains the count of reassignment for the problem for example: 1,2,3,4
[ReopenCount] -- Contains the count of reopen for the problem for example: 1,2,3,4
[ResolvedDateTime] -- Contains the date and time when problem is resolved for example: 2022-01-01 00:00:00.000
[RCAStartTime] -- Contains the date and time when problem RCA started for example: 2022-01-01 00:00:00.000
[RCAEndTime] -- Contains the date and time when problem RCA ended for example: 2022-01-01 00:00:00.000
[MadeSLA] -- Contains the date and time when problem made SLA for example: 2022-01-01 00:00:00.000
[RelatedIncidents] -- Contains the count of related incidents for the problem
[JiraId] -- Contains the Jira ID of the problem for example: BTIOOM-3707,BTCRE-2129
[Impact] -- Contains the impact of the problem for example: High, Medium, Low

INSTRUCTIONS:
1. PLEASE DISPLAY EVERYTHING IN DAYS AND HOURS
2. Don't use 'LIMIT' keyword in the query instead use 'TOP' keyword for SQL Server
3. Before generating the SQL Query please make sure for checking the presence of Data in the tables and if the data is not present then please provide the response as "Sorry, I could not find the information you are looking for. Please try again with a different question."
4. If there is any data present in common in both tables, please join the tables and provide the response.
5. USE THE CAST OR CONVERT FUNCTION TO CONVERT THE DATE AND TIME TO THE REQUIRED FORMAT
6. If you're using any single quotes in the query, please use double quotes instead.
7. USE CAST OR CONVERT FUNCTION TO Convert THE VARCHAR TO BIGINT
8. EXTRACT THE SQL QUERY FROM THE RESPONSE AND EXECUTE THE QUERY IN THE SQL SERVER TO FETCH THE DATA
9. For boolean values use 1 for True and 0 for False

VERY IMPORTANT INSTRUCTIONS: Please generate an SQL query that captures the user's intent. DISPLAY
the USER RESPONSE IN NATURAL LANGUAGE FOR THE SAME.
USE CONTEXTUAL KNOWLEDGE, SEMANTIC KNOWLEDGE TO UNDERSTAND USER QUERY TO GENERATE RESPONSES
Consider synonyms and related terms if the user's query includes terms like 'number of', 'count', 'total,' and 'records'.
"""
        
        return f"{system_prompt}\n\nUser Question: {question}\nAI: "
    
    def update_conversation_context(self, conversation_history: List[tuple], max_history: int = 3):
        """Update the agent's conversation context"""
        if not conversation_history:
            return
        
        # Get the last few conversations for context
        recent_history = conversation_history[-max_history:]
        
        # Format the history for the memory
        formatted_history = []
        for question, response in recent_history:
            formatted_history.append(f"Human: {question}")
            formatted_history.append(f"AI: {response}")
        
        # Update memory with recent context
        context = "\n".join(formatted_history)
        self.memory.chat_memory.add_user_message(f"Previous conversation context: {context}")


# Global service instances
database_service = DatabaseService()
llm_service = LLMService()
sql_agent_service = SQLAgentService(database_service, llm_service)