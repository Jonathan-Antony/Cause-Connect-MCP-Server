#!/usr/bin/env python3
"""
MCP Server for Donor-NGO Interaction - Complete Fixed Implementation
"""

import json
import uuid
import logging
import os
import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class PledgeStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    MET = "met"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class ComplianceStatus(str, Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    REGISTERED = "registered"
    READY = "ready"
    REJECTED = "rejected"

class AgentType(str, Enum):
    DONOR = "donor"
    COMPLIANCE = "compliance"
    ADMIN = "admin"
    SPONSOR = "sponsor"
    NGO = "ngo"

# =============================================================================
# Models
# =============================================================================

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

@dataclass
class Pledge:
    id: str
    donor_id: str
    ngo_id: str
    amount: float
    target_amount: float
    status: PledgeStatus
    created_at: datetime
    expires_at: Optional[datetime]
    message: Optional[str] = None

@dataclass
class ComplianceRecord:
    ngo_id: str
    status: ComplianceStatus
    documents: List[str]
    last_checked: datetime
    notes: Optional[str] = None

@dataclass
class Sponsor:
    sponsor_id: str
    name: str
    fee_percent: float
    eligible_regions: List[str]
    min_amount: float = 1000.0

# =============================================================================
# Every.org MCP Client (Subprocess-based)
# =============================================================================

class EveryOrgMCPClient:
    """Client for Every.org MCP server via subprocess communication"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("EVERY_ORG_API_KEY", "pk_live_996a88e5cf35627e8ecd91a6e422ec64")
        self.process = None
        self.initialized = False
        self.request_id_counter = 0
        self._pending_responses = {}
        logger.info(f"EveryOrgMCPClient initialized with API key: {self.api_key[:10]}...")
    
    def _create_server_script(self) -> str:
        """Create the Every.org MCP server script with correct API endpoints"""
        current_dir = os.path.dirname(__file__)
        server_script = os.path.join(current_dir, "everyorg_mcp_server.py")
        
        # Always recreate to ensure we have the latest version
        server_code = '''#!/usr/bin/env python3
"""
Every.org MCP Server with correct API endpoints
"""
import asyncio
import sys
import json
import httpx
import os
from typing import Dict, Any, Optional

class EveryOrgMCPServer:
    def __init__(self):
        self.api_key = os.getenv("EVERY_ORG_API_KEY", "pk_live_996a88e5cf35627e8ecd91a6e422ec64")
        self.every_org_api_base = "https://partners.every.org/v0.2"
        self.initialized = False
        self.client = None
        
        sys.stderr.write(f"Server starting with API key: {self.api_key[:10]}...\\n")
        sys.stderr.flush()
        
        self.capabilities = {
            "tools": {"listChanged": False},
            "resources": {},
            "prompts": {}
        }
        
        self.tools = [
            {
                "name": "search_nonprofits",
                "description": "Search for nonprofits using Every.org API",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for nonprofits"},
                        "cause": {"type": "string", "description": "Filter by cause (optional)"},
                        "limit": {"type": "integer", "description": "Number of results", "minimum": 1, "maximum": 50, "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_nonprofit_details",
                "description": "Get detailed information about a specific nonprofit",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "nonprofit_id": {"type": "string", "description": "The Every.org nonprofit ID or EIN"}
                    },
                    "required": ["nonprofit_id"]
                }
            },
            {
                "name": "browse_nonprofits_by_cause",
                "description": "Browse nonprofits by specific cause category",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "cause": {"type": "string", "description": "Cause category to browse"},
                        "limit": {"type": "integer", "description": "Number of results", "minimum": 1, "maximum": 50, "default": 10},
                        "page": {"type": "integer", "description": "Page number", "minimum": 1, "default": 1}
                    },
                    "required": ["cause"]
                }
            }
        ]
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    def send_response(self, response: Dict[str, Any]):
        json_str = json.dumps(response)
        print(json_str, flush=True)
    
    def send_error(self, request_id: Any, code: int, message: str):
        self.send_response({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message}
        })
    
    async def handle_initialize(self, request: Dict[str, Any]):
        self.initialized = True
        response = {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": self.capabilities,
                "serverInfo": {"name": "every-org-mcp-server", "version": "1.0.0"}
            }
        }
        self.send_response(response)
    
    async def handle_tools_list(self, request: Dict[str, Any]):
        if not self.initialized:
            self.send_error(request["id"], -32002, "Server not initialized")
            return
        
        response = {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"tools": self.tools}
        }
        self.send_response(response)
    
    async def search_nonprofits_api(self, query: str, cause: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Search nonprofits using correct Every.org API format"""
        url = f"{self.every_org_api_base}/search/{query}"
        params = {"apiKey": self.api_key, "take": limit}
        
        if cause:
            params["causes"] = cause.lower()
        
        sys.stderr.write(f"Making request to: {url} with params: {params}\\n")
        sys.stderr.flush()
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_nonprofit_details_api(self, nonprofit_id: str) -> Dict[str, Any]:
        """Get nonprofit details using correct Every.org API format"""
        url = f"{self.every_org_api_base}/nonprofit/{nonprofit_id}"
        params = {"apiKey": self.api_key}
        
        sys.stderr.write(f"Making request to: {url} with params: {params}\\n")
        sys.stderr.flush()
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def browse_nonprofits_by_cause_api(self, cause: str, limit: int = 10, page: int = 1) -> Dict[str, Any]:
        """Browse nonprofits by cause using correct Every.org API format"""
        cause_lower = cause.lower()
        url = f"{self.every_org_api_base}/browse/{cause_lower}"
        params = {"apiKey": self.api_key, "take": limit, "page": page}
        
        sys.stderr.write(f"Making request to: {url} with params: {params}\\n")
        sys.stderr.flush()
        
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def handle_tools_call(self, request: Dict[str, Any]):
        if not self.initialized:
            self.send_error(request["id"], -32002, "Server not initialized")
            return
        
        try:
            params = request["params"]
            tool_name = params["name"]
            arguments = params.get("arguments", {})
            
            sys.stderr.write(f"Tool call: {tool_name} with args: {arguments}\\n")
            sys.stderr.flush()
            
            if tool_name == "search_nonprofits":
                result = await self.search_nonprofits_api(
                    arguments["query"], 
                    arguments.get("cause"), 
                    arguments.get("limit", 10)
                )
            elif tool_name == "get_nonprofit_details":
                result = await self.get_nonprofit_details_api(arguments["nonprofit_id"])
            elif tool_name == "browse_nonprofits_by_cause":
                result = await self.browse_nonprofits_by_cause_api(
                    arguments["cause"],
                    arguments.get("limit", 10),
                    arguments.get("page", 1)
                )
            else:
                self.send_error(request["id"], -32601, f"Unknown tool: {tool_name}")
                return
            
            response = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False
                }
            }
            self.send_response(response)
            
        except Exception as e:
            sys.stderr.write(f"Tool execution error: {e}\\n")
            sys.stderr.flush()
            self.send_error(request["id"], -32603, f"Tool execution failed: {str(e)}")
    
    async def handle_request(self, request: Dict[str, Any]):
        method = request.get("method")
        if method == "initialize":
            await self.handle_initialize(request)
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            await self.handle_tools_list(request)
        elif method == "tools/call":
            await self.handle_tools_call(request)
        else:
            self.send_error(request.get("id"), -32601, f"Method not found: {method}")
    
    async def run(self):
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    await self.handle_request(request)
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"JSON decode error: {e}\\n")
                except Exception as e:
                    sys.stderr.write(f"Request handling error: {e}\\n")
        except KeyboardInterrupt:
            pass

async def main():
    async with EveryOrgMCPServer() as server:
        await server.run()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Write the server script
        with open(server_script, 'w') as f:
            f.write(server_code)
        os.chmod(server_script, 0o755)
        
        logger.info(f"Created Every.org MCP server script at: {server_script}")
        return server_script
    
    async def start(self):
        """Start the Every.org MCP server subprocess"""
        if self.process:
            logger.warning("Server process already running")
            return
        
        try:
            # Create the server script
            server_script = self._create_server_script()
            
            # Set environment variable for API key
            env = os.environ.copy()
            env["EVERY_ORG_API_KEY"] = self.api_key
            
            # Start the subprocess
            self.process = await asyncio.create_subprocess_exec(
                "python3", server_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Start the response reader task
            asyncio.create_task(self._read_responses())
            
            # Wait a moment for process to start
            await asyncio.sleep(0.5)
            
            # Initialize the server
            await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "donor-ngo-mcp-server", "version": "1.0.0"}
            })
            
            # Send initialized notification
            await self._send_notification("notifications/initialized", {})
            
            self.initialized = True
            logger.info("Every.org MCP server started and initialized")
            
        except Exception as e:
            logger.error(f"Failed to start Every.org MCP server: {e}")
            raise
    
    async def stop(self):
        """Stop the Every.org MCP server subprocess"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.process = None
            self.initialized = False
            logger.info("Every.org MCP server stopped")
    
    async def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response"""
        if not self.process:
            raise Exception("Server not started")
        
        request_id = self.request_id_counter
        self.request_id_counter += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        # Create future for response
        future = asyncio.Future()
        self._pending_responses[request_id] = future
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Wait for response with timeout
        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_responses.pop(request_id, None)
            raise Exception("Request timeout")
    
    async def _send_notification(self, method: str, params: Dict[str, Any] = None):
        """Send a JSON-RPC notification (no response expected)"""
        if not self.process:
            raise Exception("Server not started")
        
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        
        notification_json = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_json.encode())
        await self.process.stdin.drain()
    
    async def _read_responses(self):
        """Read responses from the server subprocess"""
        if not self.process:
            return
        
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                try:
                    response = json.loads(line.decode().strip())
                    request_id = response.get("id")
                    
                    if request_id is not None and request_id in self._pending_responses:
                        future = self._pending_responses.pop(request_id)
                        if "error" in response:
                            future.set_exception(Exception(response["error"]["message"]))
                        else:
                            future.set_result(response["result"])
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                    
        except Exception as e:
            logger.error(f"Error reading responses: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from Every.org server"""
        if not self.initialized:
            raise Exception("Client not initialized")
        
        result = await self._send_request("tools/list")
        return result.get("tools", [])
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the Every.org server"""
        if not self.initialized:
            raise Exception("Client not initialized")
        
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        # Extract the actual result from MCP response format
        if "content" in result and isinstance(result["content"], list):
            for content_item in result["content"]:
                if content_item.get("type") == "text":
                    try:
                        return json.loads(content_item["text"])
                    except json.JSONDecodeError:
                        return {"raw_text": content_item["text"]}
        
        return result

# =============================================================================
# Database
# =============================================================================

class MockDatabase:
    def __init__(self):
        self.pledges: Dict[str, Pledge] = {}
        self.compliance_records: Dict[str, ComplianceRecord] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.sponsors: List[Sponsor] = [
            Sponsor(
                sponsor_id="spon-001",
                name="Tides Foundation",
                fee_percent=8.0,
                eligible_regions=["US", "Global"],
                min_amount=5000.0
            ),
            Sponsor(
                sponsor_id="spon-002",
                name="Open Collective Foundation",
                fee_percent=5.0,
                eligible_regions=["Global"],
                min_amount=1000.0
            ),
            Sponsor(
                sponsor_id="spon-003",
                name="Donor Advised Fund",
                fee_percent=3.0,
                eligible_regions=["US", "Canada"],
                min_amount=10000.0
            )
        ]
        self._init_mock_compliance_data()
    
    def _init_mock_compliance_data(self):
        mock_compliance = [
            ComplianceRecord(
                ngo_id="proj-001",
                status=ComplianceStatus.PENDING,
                documents=["ID", "Project Plan"],
                last_checked=datetime.now() - timedelta(hours=1),
                notes="Initial submission received"
            ),
            ComplianceRecord(
                ngo_id="proj-002",
                status=ComplianceStatus.REGISTERED,
                documents=["ID", "Project Plan", "Tax Clearance"],
                last_checked=datetime.now() - timedelta(hours=2),
                notes="All documents verified"
            ),
            ComplianceRecord(
                ngo_id="proj-003",
                status=ComplianceStatus.UNDER_REVIEW,
                documents=["ID", "Project Plan", "Financial Statement"],
                last_checked=datetime.now() - timedelta(minutes=30),
                notes="Reviewing financial documentation"
            )
        ]
        
        for record in mock_compliance:
            self.compliance_records[record.ngo_id] = record
    
    def log_action(self, agent_type: str, action: str, details: Dict[str, Any]):
        """Log all actions for audit trail"""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "action": action,
            "details": details
        })

# =============================================================================
# Auth Service
# =============================================================================

class AuthService:
    @staticmethod
    def verify_token(token: Optional[str]) -> Dict[str, Any]:
        auth_mode = os.getenv("AUTH_MODE", "AUTO")
        
        if auth_mode == "STRICT" and not token:
            raise HTTPException(status_code=401, detail="No authorization token provided")
        
        if token and token.startswith("Bearer "):
            token = token[7:]
        
        if token:
            if token.startswith("donor_"):
                return {"agent_type": AgentType.DONOR, "user_id": token.split("_")[1]}
            elif token.startswith("ngo_"):
                parts = token.split("_")
                if len(parts) >= 3:
                    ngo_id = parts[1]
                    user_id = "_".join(parts[2:])
                    return {"agent_type": AgentType.NGO, "user_id": user_id, "ngo_id": ngo_id}
                else:
                    return {"agent_type": AgentType.NGO, "user_id": parts[1], "ngo_id": "proj-001"}
            elif token.startswith("compliance_"):
                return {"agent_type": AgentType.COMPLIANCE, "user_id": token.split("_")[1]}
            elif token.startswith("admin_"):
                return {"agent_type": AgentType.ADMIN, "user_id": token.split("_")[1]}
            elif token.startswith("sponsor_"):
                return {"agent_type": AgentType.SPONSOR, "user_id": token.split("_")[1]}
            else:
                if auth_mode == "AUTO":
                    logger.info(f"Unknown token format, defaulting to admin: {token[:20]}...")
                    return {"agent_type": AgentType.ADMIN, "user_id": "auto_admin"}
                else:
                    raise HTTPException(status_code=401, detail="Invalid authorization token")
        else:
            if auth_mode == "AUTO":
                logger.info("No auth token provided, auto-assigning admin role for testing")
                return {"agent_type": AgentType.ADMIN, "user_id": "auto_admin"}
            else:
                raise HTTPException(status_code=401, detail="No authorization token provided")
    
    @staticmethod
    def check_tool_access(agent_type: AgentType, tool_name: str) -> bool:
        access_rules = {
            AgentType.DONOR: ["ngo_info", "pledge", "compliance", "sponsor_matching", "payment", "ngo_analytics", 
                             "search_nonprofits", "get_nonprofit_details", "browse_nonprofits_by_cause"],
            AgentType.NGO: ["ngo_info", "pledge", "compliance", "sponsor_matching", "payment", 
                           "search_nonprofits", "get_nonprofit_details", "browse_nonprofits_by_cause"],
            AgentType.COMPLIANCE: ["compliance", "sponsor_matching", "ngo_info", 
                                  "search_nonprofits", "get_nonprofit_details", "browse_nonprofits_by_cause"],
            AgentType.ADMIN: ["ngo_info", "pledge", "compliance", "sponsor_matching", "payment", "ngo_analytics",
                             "search_nonprofits", "get_nonprofit_details", "browse_nonprofits_by_cause"],
            AgentType.SPONSOR: ["ngo_info", "compliance", "sponsor_matching", 
                               "search_nonprofits", "get_nonprofit_details", "browse_nonprofits_by_cause"]
        }
        return tool_name in access_rules.get(agent_type, [])

def get_current_agent(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    return AuthService.verify_token(authorization)

# =============================================================================
# Global instances
# =============================================================================

db = MockDatabase()
everyorg_client = None

# =============================================================================
# Tool Handlers
# =============================================================================

async def handle_ngo_info_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy ngo_info tool - redirects to Every.org tools"""
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "ngo_info"):
        raise HTTPException(status_code=403, detail="Access denied to NGO Info tool")
    
    # Redirect to appropriate Every.org tool based on parameters
    if params.get("ngo_id"):
        return await handle_get_nonprofit_details_tool({"nonprofit_id": params["ngo_id"]}, agent)
    elif params.get("category"):
        return await handle_browse_nonprofits_by_cause_tool({
            "cause": params["category"],
            "limit": params.get("limit", 10)
        }, agent)
    elif params.get("name"):
        return await handle_search_nonprofits_tool({
            "query": params["name"],
            "limit": params.get("limit", 10)
        }, agent)
    else:
        # Default search
        return await handle_search_nonprofits_tool({
            "query": "charity",
            "limit": params.get("limit", 10)
        }, agent)

async def handle_search_nonprofits_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    """Passthrough tool for Every.org search_nonprofits"""
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "search_nonprofits"):
        raise HTTPException(status_code=403, detail="Access denied to Search Nonprofits tool")
    
    if not everyorg_client or not everyorg_client.initialized:
        raise HTTPException(status_code=503, detail="Every.org service not available")
    
    try:
        result = await everyorg_client.call_tool("search_nonprofits", params)
        
        # Log action
        db.log_action(agent["agent_type"], "search_nonprofits", {
            "user_id": agent["user_id"],
            "params": params
        })
        
        # Add agent context
        result["agent_context"] = {
            "type": agent["agent_type"],
            "scoped_to_ngo": agent.get("ngo_id") if agent["agent_type"] == AgentType.NGO else None,
            "source": "every_org_mcp"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calling Every.org search_nonprofits: {e}")
        raise HTTPException(status_code=500, detail=f"Every.org API error: {str(e)}")

async def handle_get_nonprofit_details_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    """Passthrough tool for Every.org get_nonprofit_details"""
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "get_nonprofit_details"):
        raise HTTPException(status_code=403, detail="Access denied to Get Nonprofit Details tool")
    
    if not everyorg_client or not everyorg_client.initialized:
        raise HTTPException(status_code=503, detail="Every.org service not available")
    
    try:
        result = await everyorg_client.call_tool("get_nonprofit_details", params)
        
        # Log action
        db.log_action(agent["agent_type"], "get_nonprofit_details", {
            "user_id": agent["user_id"],
            "params": params
        })
        
        # Add agent context
        result["agent_context"] = {
            "type": agent["agent_type"],
            "scoped_to_ngo": agent.get("ngo_id") if agent["agent_type"] == AgentType.NGO else None,
            "source": "every_org_mcp"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calling Every.org get_nonprofit_details: {e}")
        raise HTTPException(status_code=500, detail=f"Every.org API error: {str(e)}")

async def handle_browse_nonprofits_by_cause_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    """Passthrough tool for Every.org browse_nonprofits_by_cause"""
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "browse_nonprofits_by_cause"):
        raise HTTPException(status_code=403, detail="Access denied to Browse Nonprofits by Cause tool")
    
    if not everyorg_client or not everyorg_client.initialized:
        raise HTTPException(status_code=503, detail="Every.org service not available")
    
    try:
        result = await everyorg_client.call_tool("browse_nonprofits_by_cause", params)
        
        # Log action
        db.log_action(agent["agent_type"], "browse_nonprofits_by_cause", {
            "user_id": agent["user_id"],
            "params": params
        })
        
        # Add agent context
        result["agent_context"] = {
            "type": agent["agent_type"],
            "scoped_to_ngo": agent.get("ngo_id") if agent["agent_type"] == AgentType.NGO else None,
            "source": "every_org_mcp"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calling Every.org browse_nonprofits_by_cause: {e}")
        raise HTTPException(status_code=500, detail=f"Every.org API error: {str(e)}")

async def handle_pledge_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "pledge"):
        raise HTTPException(status_code=403, detail="Access denied to Pledge tool")
    
    action = params.get("action", "create")
    
    if action == "create":
        if agent["agent_type"] not in [AgentType.DONOR, AgentType.ADMIN]:
            raise HTTPException(status_code=403, detail="Only Donor agents can create pledges")
        
        pledge_id = str(uuid.uuid4())
        pledge = Pledge(
            id=pledge_id,
            donor_id=agent["user_id"],
            ngo_id=params.get("ngo_id", "proj-001"),
            amount=params.get("amount", 1000.0),
            target_amount=params.get("target_amount", 25000.0),
            status=PledgeStatus.ACTIVE,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=30),
            message=params.get("message")
        )
        
        db.pledges[pledge_id] = pledge
        
        db.log_action(agent["agent_type"], "pledge_created", {
            "user_id": agent["user_id"],
            "pledge_id": pledge_id,
            "ngo_id": params.get("ngo_id"),
            "amount": params.get("amount")
        })
        
        return {"pledge": asdict(pledge), "message": "Pledge created successfully"}
    
    elif action == "list":
        pledges = list(db.pledges.values())
        
        if agent["agent_type"] == AgentType.DONOR:
            pledges = [p for p in pledges if p.donor_id == agent["user_id"]]
        elif agent["agent_type"] == AgentType.NGO:
            ngo_id = agent.get("ngo_id")
            if ngo_id:
                pledges = [p for p in pledges if p.ngo_id == ngo_id]
            else:
                pledges = []
        elif agent["agent_type"] != AgentType.ADMIN:
            pledges = []
        
        return {
            "pledges": [asdict(p) for p in pledges],
            "count": len(pledges)
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown pledge action: {action}")

async def handle_compliance_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "compliance"):
        raise HTTPException(status_code=403, detail="Access denied to Compliance tool")
    
    action = params.get("action", "status")
    
    if action == "status":
        ngo_id = params.get("ngo_id")
        
        if agent["agent_type"] == AgentType.NGO:
            ngo_id = agent.get("ngo_id")
            if not ngo_id:
                raise HTTPException(status_code=400, detail="NGO agent missing ngo_id")
        
        if not ngo_id:
            if agent["agent_type"] not in [AgentType.ADMIN, AgentType.COMPLIANCE]:
                raise HTTPException(status_code=403, detail="Cannot view all compliance records")
            
            records = [asdict(record) for record in db.compliance_records.values()]
            return {"compliance_records": records, "count": len(records)}
        
        if ngo_id not in db.compliance_records:
            raise HTTPException(status_code=404, detail="Compliance record not found")
        
        record = db.compliance_records[ngo_id]
        return {"compliance_record": asdict(record)}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown compliance action: {action}")

async def handle_sponsor_matching_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "sponsor_matching"):
        raise HTTPException(status_code=403, detail="Access denied to Sponsor Matching tool")
    
    action = params.get("action", "match")
    
    if action == "match":
        region = params.get("region", "Global")
        amount = params.get("amount", 0)
        
        eligible_sponsors = []
        for sponsor in db.sponsors:
            if (region in sponsor.eligible_regions or "Global" in sponsor.eligible_regions) and \
               amount >= sponsor.min_amount:
                eligible_sponsors.append(asdict(sponsor))
        
        eligible_sponsors.sort(key=lambda x: x["fee_percent"])
        
        return {
            "eligible_sponsors": eligible_sponsors,
            "count": len(eligible_sponsors),
            "recommended": eligible_sponsors[0] if eligible_sponsors else None
        }
        
    elif action == "list":
        sponsors = [asdict(sponsor) for sponsor in db.sponsors]
        return {"sponsors": sponsors, "count": len(sponsors)}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown sponsor matching action: {action}")

async def handle_payment_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "payment"):
        raise HTTPException(status_code=403, detail="Access denied to Payment tool")
    
    action = params.get("action", "simulate")
    
    if action == "simulate":
        pledge_id = params.get("pledge_id")
        sponsor_id = params.get("sponsor_id", "spon-002")
        
        if not pledge_id or pledge_id not in db.pledges:
            raise HTTPException(status_code=404, detail="Pledge not found")
        
        pledge = db.pledges[pledge_id]
        sponsor = next((s for s in db.sponsors if s.sponsor_id == sponsor_id), None)
        if not sponsor:
            raise HTTPException(status_code=404, detail="Sponsor not found")
        
        total_amount = pledge.amount
        sponsor_fee = total_amount * (sponsor.fee_percent / 100)
        disbursement_amount = total_amount - sponsor_fee
        
        pledge.status = PledgeStatus.MET
        
        payment_simulation = {
            "payment_id": str(uuid.uuid4()),
            "pledge_id": pledge_id,
            "ngo_id": pledge.ngo_id,
            "total_amount": total_amount,
            "sponsor_fee": sponsor_fee,
            "disbursement_amount": disbursement_amount,
            "sponsor": sponsor.name,
            "processed_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        return {
            "payment_result": payment_simulation,
            "message": "Payment simulation completed successfully"
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown payment action: {action}")

async def handle_ngo_analytics_tool(params: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
    if not AuthService.check_tool_access(AgentType(agent["agent_type"]), "ngo_analytics"):
        raise HTTPException(status_code=403, detail="Access denied to NGO Analytics tool")
    
    if agent["agent_type"] not in [AgentType.DONOR, AgentType.ADMIN]:
        raise HTTPException(status_code=403, detail="Only Donor agents can access NGO analytics")
    
    action = params.get("action", "sector_summary")
    
    if action == "sector_summary":
        return {
            "sector_analysis": {
                "education": {"ngo_count": 45, "avg_funding": 25000},
                "health": {"ngo_count": 38, "avg_funding": 35000},
                "environment": {"ngo_count": 22, "avg_funding": 18000}
            },
            "total_ngos_analyzed": 105
        }
    elif action == "top_funded":
        limit = params.get("limit", 10)
        mock_ngos = [
            {"id": "proj-001", "name": "Clean Water Initiative", "funding": 25000},
            {"id": "proj-002", "name": "Education for All", "funding": 75000},
            {"id": "proj-003", "name": "Healthcare Access", "funding": 50000}
        ][:limit]
        return {
            "top_organizations": mock_ngos,
            "count": len(mock_ngos)
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown analytics action: {action}")

# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global everyorg_client
    
    logger.info("Starting MCP Donor-NGO Server...")
    logger.info(f"Authentication mode: {os.getenv('AUTH_MODE', 'AUTO')}")
    
    # Initialize Every.org MCP client
    everyorg_client = EveryOrgMCPClient()
    
    try:
        await everyorg_client.start()
        logger.info("Every.org MCP client started successfully")
        
        # Test the client by listing available tools
        tools = await everyorg_client.list_tools()
        logger.info(f"Every.org MCP server offers {len(tools)} tools: {[t['name'] for t in tools]}")
        
    except Exception as e:
        logger.error(f"Failed to start Every.org MCP client: {e}")
        # Continue without the client - tools will return service unavailable errors
    
    logger.info("Server ready")
    
    yield
    
    # Shutdown
    if everyorg_client:
        await everyorg_client.stop()
    logger.info("Shutting down MCP Donor-NGO Server...")

app = FastAPI(
    title="MCP Donor-NGO Server",
    description="MCP server for donor-NGO interactions with Every.org integration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MCP Endpoints
# =============================================================================

def create_success_response(result: Any, request_id: Optional[Union[str, int]]) -> JSONResponse:
    return JSONResponse(content={
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id
    })

def create_error_response(code: int, message: str, request_id: Optional[Union[str, int]], data: Optional[Any] = None) -> JSONResponse:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    
    return JSONResponse(content={
        "jsonrpc": "2.0", 
        "error": error,
        "id": request_id
    })

@app.post("/mcp")
async def mcp_handler(request: JSONRPCRequest, agent: Dict[str, Any] = Depends(get_current_agent)):
    try:
        method = request.method
        params = request.params or {}
        request_id = request.id
        
        if agent["user_id"] == "auto_admin":
            logger.info(f"Auto-authenticated JSON-RPC request: method={method}, agent_type={agent['agent_type']}")
        
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "logging": {},
                    "prompts": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "donor-ngo-mcp-server",
                    "version": "1.0.0"
                }
            }
            return create_success_response(result, request_id)
        
        elif method.startswith("notifications/"):
            logger.info(f"Received notification: {method}")
            return JSONResponse(content={}, status_code=200)
        
        elif method == "tools/list":
            # Get tools from Every.org server if available
            everyorg_tools = []
            if everyorg_client and everyorg_client.initialized:
                try:
                    everyorg_tools = await everyorg_client.list_tools()
                except Exception as e:
                    logger.warning(f"Failed to get tools from Every.org server: {e}")
            
            # Define our core tools
            core_tools = [
                {
                    "name": "ngo_info",
                    "description": "Legacy NGO info tool (redirects to Every.org tools)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "NGO name to search for"},
                            "category": {"type": "string", "description": "Category/cause to filter by"},
                            "ngo_id": {"type": "string", "description": "Specific NGO ID to get details for"},
                            "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                        }
                    }
                },
                {
                    "name": "pledge",
                    "description": "Manage donation pledges",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["create", "list"]},
                            "ngo_id": {"type": "string"},
                            "amount": {"type": "number"},
                            "target_amount": {"type": "number"},
                            "message": {"type": "string"}
                        },
                        "required": ["action"]
                    }
                },
                {
                    "name": "compliance",
                    "description": "View NGO compliance status",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["status"]},
                            "ngo_id": {"type": "string"}
                        },
                        "required": ["action"]
                    }
                },
                {
                    "name": "sponsor_matching",
                    "description": "Match NGOs with sponsors",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["match", "list"]},
                            "region": {"type": "string", "default": "Global"},
                            "amount": {"type": "number"}
                        },
                        "required": ["action"]
                    }
                },
                {
                    "name": "payment",
                    "description": "Simulate payment processing",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["simulate"]},
                            "pledge_id": {"type": "string"},
                            "sponsor_id": {"type": "string", "default": "spon-002"}
                        },
                        "required": ["action", "pledge_id"]
                    }
                },
                {
                    "name": "ngo_analytics",
                    "description": "Analytics and insights across NGOs",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["sector_summary", "top_funded"]},
                            "limit": {"type": "integer", "default": 10}
                        },
                        "required": ["action"]
                    }
                }
            ]
            
            # Combine core tools with Every.org tools
            all_tools = core_tools + everyorg_tools
            
            result = {"tools": all_tools}
            return create_success_response(result, request_id)
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            # Route to appropriate handler
            if tool_name == "ngo_info":
                tool_result = await handle_ngo_info_tool(arguments, agent)
            elif tool_name == "search_nonprofits":
                tool_result = await handle_search_nonprofits_tool(arguments, agent)
            elif tool_name == "get_nonprofit_details":
                tool_result = await handle_get_nonprofit_details_tool(arguments, agent)
            elif tool_name == "browse_nonprofits_by_cause":
                tool_result = await handle_browse_nonprofits_by_cause_tool(arguments, agent)
            elif tool_name == "pledge":
                tool_result = await handle_pledge_tool(arguments, agent)
            elif tool_name == "compliance":
                tool_result = await handle_compliance_tool(arguments, agent)
            elif tool_name == "sponsor_matching":
                tool_result = await handle_sponsor_matching_tool(arguments, agent)
            elif tool_name == "payment":
                tool_result = await handle_payment_tool(arguments, agent)
            elif tool_name == "ngo_analytics":
                tool_result = await handle_ngo_analytics_tool(arguments, agent)
            else:
                return create_error_response(-32601, f"Unknown tool: {tool_name}", request_id)
            
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(tool_result, indent=2, default=str)
                    }
                ]
            }
            return create_success_response(result, request_id)
        
        else:
            return create_error_response(-32601, f"Unknown method: {method}", request_id)
        
    except HTTPException as e:
        return create_error_response(e.status_code, e.detail, getattr(request, 'id', None))
    except Exception as e:
        logger.error(f"Error processing JSON-RPC request: {e}")
        return create_error_response(-32603, f"Internal error: {str(e)}", getattr(request, 'id', None))

# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    # Check Every.org client status
    everyorg_status = "disconnected"
    everyorg_tools_count = 0
    
    if everyorg_client:
        if everyorg_client.initialized:
            everyorg_status = "connected"
            try:
                tools = await everyorg_client.list_tools()
                everyorg_tools_count = len(tools)
            except:
                everyorg_status = "connected_but_tools_unavailable"
        elif everyorg_client.process:
            everyorg_status = "starting"
        else:
            everyorg_status = "not_started"
    else:
        everyorg_status = "not_initialized"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "auth_mode": os.getenv("AUTH_MODE", "AUTO"),
        "everyorg_mcp_status": everyorg_status,
        "everyorg_tools_available": everyorg_tools_count
    }

@app.get("/ping")
async def ping():
    return {
        "message": "pong", 
        "timestamp": datetime.now().isoformat(),
        "auth_mode": os.getenv("AUTH_MODE", "AUTO")
    }

@app.get("/everyorg/status")
async def everyorg_status():
    """Get detailed Every.org MCP client status"""
    if not everyorg_client:
        return {
            "status": "unavailable",
            "reason": "client_not_initialized"
        }
    
    status = {
        "client_created": True,
        "process_running": bool(everyorg_client.process and everyorg_client.process.returncode is None),
        "initialized": everyorg_client.initialized,
        "api_key_configured": bool(everyorg_client.api_key),
        "api_key_prefix": everyorg_client.api_key[:10] if everyorg_client.api_key else None
    }
    
    if everyorg_client.initialized:
        try:
            tools = await everyorg_client.list_tools()
            status["tools_available"] = len(tools)
            status["tool_names"] = [t["name"] for t in tools]
        except Exception as e:
            status["tools_error"] = str(e)
    
    return status

@app.get("/test-everyorg")
async def test_everyorg_connection():
    """Test endpoint to verify Every.org MCP connection"""
    if not everyorg_client:
        return {"error": "Every.org client not initialized"}
    
    if not everyorg_client.initialized:
        return {"error": "Every.org client not connected"}
    
    try:
        # Test search_nonprofits tool
        result = await everyorg_client.call_tool("search_nonprofits", {
            "query": "education",
            "limit": 1
        })
        
        return {
            "test": "search_nonprofits",
            "success": True,
            "result_keys": list(result.keys()) if isinstance(result, dict) else [],
            "sample_result": result
        }
            
    except Exception as e:
        return {
            "test": "search_nonprofits",
            "success": False,
            "error": str(e)
        }

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCP Donor-NGO Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )