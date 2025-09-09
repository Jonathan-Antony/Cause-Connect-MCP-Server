# Cause Connect MCP Server

A Model Context Protocol (MCP) server that facilitates donor-NGO interactions with integrated Every.org nonprofit data access. This server provides tools for nonprofit discovery, donation management, compliance tracking, sponsor matching, and analytics.

## Features

- **Nonprofit Data Integration**: Real-time access to Every.org's database of 1M+ nonprofits
- **Role-Based Access Control**: Different permissions for donors, NGOs, compliance officers, sponsors, and admins
- **Donation Management**: Pledge creation and tracking system
- **Compliance Monitoring**: NGO verification and document tracking
- **Sponsor Matching**: Automated matching of nonprofits with appropriate sponsors
- **Payment Simulation**: Mock payment processing with sponsor fee calculations
- **Analytics Dashboard**: Sector analysis and funding insights

## Architecture

The server uses a hybrid architecture:
- **Main Server** (FastAPI): Handles MCP protocol and core business logic
- **Every.org Subprocess**: Dedicated MCP server for nonprofit data API calls
- **Communication**: JSON-RPC over stdin/stdout between processes

## Available Tools

### Core Tools

1. **ngo_info** - Legacy tool that redirects to appropriate Every.org tools
2. **pledge** - Create and manage donation pledges
3. **compliance** - View NGO compliance status and documents
4. **sponsor_matching** - Match NGOs with eligible sponsors
5. **payment** - Simulate payment processing with fee calculations
6. **ngo_analytics** - Access sector analysis and funding insights

### Every.org Integration Tools

7. **search_nonprofits** - Search nonprofits by name or keywords
8. **get_nonprofit_details** - Get detailed information about specific nonprofits
9. **browse_nonprofits_by_cause** - Browse nonprofits by cause category (education, health, environment, etc.)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (recommended)
- Every.org API key

### Docker Setup (Recommended)

1. **Build the container:**
   ```bash
   docker build -t donor-ngo-mcp-server .
   ```

2. **Run with environment variables:**
   ```bash
   docker run -p 8003:8003 \
     -e EVERY_ORG_API_KEY=your_api_key_here \
     -e AUTH_MODE=AUTO \
     donor-ngo-mcp-server
   ```

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export EVERY_ORG_API_KEY=your_api_key_here
   export AUTH_MODE=AUTO
   ```

3. **Run the server:**
   ```bash
   python main.py --host 127.0.0.1 --port 8003
   ```

## Authentication & Authorization

### Authentication Modes

- **AUTO** (default): Auto-assigns admin role for testing
- **STRICT**: Requires valid authorization tokens

### Token Format

- **Donors**: `donor_123`
- **NGOs**: `ngo_org123_user456`
- **Compliance**: `compliance_123`
- **Sponsors**: `sponsor_123`
- **Admins**: `admin_123`

### Access Control

Different agent types have access to different tools:

| Tool | Donor | NGO | Compliance | Sponsor | Admin |
|------|-------|-----|------------|---------|-------|
| ngo_info | ✓ | ✓ | ✓ | ✓ | ✓ |
| search_nonprofits | ✓ | ✓ | ✓ | ✓ | ✓ |
| get_nonprofit_details | ✓ | ✓ | ✓ | ✓ | ✓ |
| browse_nonprofits_by_cause | ✓ | ✓ | ✓ | ✓ | ✓ |
| pledge | ✓ | ✓ | | | ✓ |
| compliance | ✓ | ✓ | ✓ | ✓ | ✓ |
| sponsor_matching | ✓ | ✓ | ✓ | ✓ | ✓ |
| payment | ✓ | ✓ | | | ✓ |
| ngo_analytics | ✓ | | | | ✓ |

## API Examples

### Search for Education Nonprofits

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "browse_nonprofits_by_cause",
    "arguments": {
      "cause": "education",
      "limit": 5
    }
  },
  "id": 1
}
```

### Create a Donation Pledge

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "pledge",
    "arguments": {
      "action": "create",
      "ngo_id": "some-nonprofit-id",
      "amount": 1000.0,
      "message": "Supporting education initiatives"
    }
  },
  "id": 2
}
```

### Get Nonprofit Details

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_nonprofit_details",
    "arguments": {
      "nonprofit_id": "givedirectly"
    }
  },
  "id": 3
}
```

## Health Endpoints

- **GET /health** - Overall server health status
- **GET /ping** - Simple connectivity test  
- **GET /everyorg/status** - Every.org integration status
- **GET /test-everyorg** - Test Every.org API connection

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EVERY_ORG_API_KEY` | Every.org API key | Required |
| `AUTH_MODE` | Authentication mode (AUTO/STRICT) | AUTO |
| `LOG_LEVEL` | Logging level | info |

## Supported Nonprofit Causes

The Every.org integration supports these cause categories:

- `education` - Educational institutions and programs
- `health` - Healthcare and medical research
- `environment` - Environmental conservation
- `poverty` - Poverty alleviation and social services
- `animals` - Animal welfare and protection
- `arts` - Arts, culture, and humanities
- `community` - Community development
- `disaster-relief` - Emergency response and disaster relief
- `human-services` - Social and human services
- `international` - International development
- `research` - Scientific and academic research
- `religion` - Religious organizations

## Error Handling

The server provides detailed error responses for common scenarios:

- **403 Forbidden**: Insufficient permissions for requested tool
- **404 Not Found**: Resource (pledge, NGO, etc.) not found
- **503 Service Unavailable**: Every.org integration not available
- **400 Bad Request**: Invalid parameters or malformed requests

## Development

### Project Structure

```
├── main.py                 # Main server implementation
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── everyorg_mcp_server.py # Auto-generated Every.org subprocess
└── README.md             # This file
```

### Adding New Tools

1. Create a handler function in the tool handlers section
2. Add tool definition to the `tools/list` response
3. Add routing logic in the `tools/call` handler
4. Update access control rules in `AuthService.check_tool_access()`

### Debugging

The server includes comprehensive logging. For debugging Every.org integration issues:

1. Check `/everyorg/status` endpoint for subprocess status
2. Use `/test-everyorg` to test API connectivity
3. Enable debug logging with `--log-level debug`

## License

This project is provided as-is for demonstration purposes.

## Support

For issues related to:
- **Every.org API**: Contact Every.org support
- **MCP Protocol**: See Model Context Protocol documentation
- **Server Issues**: Check the health endpoints and logs
