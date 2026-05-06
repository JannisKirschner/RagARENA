# VPN Setup Guide

To access AcmeCorp's internal network remotely, you must use the corporate VPN.

## Installation

1. Download GlobalProtect from the Software Center on your laptop
2. Open GlobalProtect and enter the portal address: vpn.acmecorp.com
3. Sign in with your AcmeCorp credentials (same as your email login)
4. Click Connect

## Troubleshooting

- If you see "Portal unreachable," check that you have internet connectivity first
- If your credentials are rejected, reset your password at https://password.acmecorp.internal
- For persistent issues, contact the IT Helpdesk via the support portal

## Split Tunneling

The VPN uses split tunneling by default. Internal resources (*.acmecorp.internal) route through the VPN; all other traffic goes directly to the internet.
