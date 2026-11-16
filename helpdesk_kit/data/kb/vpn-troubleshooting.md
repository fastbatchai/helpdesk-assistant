---
title: VPN Troubleshooting and Error Codes
department: IT Infrastructure
updated: 2026-08-22
audience: all
---

# VPN Troubleshooting and Error Codes

Try these steps first if NorthLink will not connect or keeps dropping:

1. Check that ordinary websites load. If they do not, the problem is your internet connection and not the VPN.
2. Quit NorthLink fully, then open it again.
3. Restart your laptop.
4. Check Company Apps for a NorthLink update. Versions older than 6.0 are not supported.

## Error codes

**NGL-809: certificate expired.** Your laptop's device certificate has expired, usually because the laptop has been away from the network for more than 60 days. You cannot fix this yourself. Connect to an office network and restart, or contact the help desk for a renewal.

**NGL-412: MFA timeout.** The approval request was not answered within 60 seconds. Connect again and approve the request promptly. Check your phone has a data connection.

**NGL-117: session dropped by client.** The connection was lost on your side. This commonly appears as repeated disconnects on Wi-Fi. Turn off the power-saving setting on your wireless adapter and update NorthLink. If the disconnects continue, open a ticket and attach the log from Help > Export Logs.

**NGL-530: route conflict.** Another network, often a home router or a second VPN, is using the same address range. Disconnect any other VPN. If you are at home, restart your router.

**NGL-221: account not permitted.** Your account is not in the VPN access group. This is common in a contractor's first week. Your manager must request access through the portal.

## Slow connections

All traffic passes through the tunnel, so speed depends on your own connection. Disconnect from the VPN for video calls. When working abroad, also read the Travel and Roaming guide.

## When to contact us

Open a ticket if a code above keeps returning or is not listed. Include the code, time and your location. If a whole office cannot connect, phone the help desk, because it is treated as a P1 incident.
