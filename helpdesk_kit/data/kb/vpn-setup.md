---
title: VPN Setup Guide
department: IT Infrastructure
updated: 2026-06-10
audience: all
---

# VPN Setup Guide

NorthLink is the company VPN. You need it to reach shared drives, the warehouse management system and internal sites when you are not on an office network. You do not need it for email, Microsoft Teams or the help-desk portal.

## Before you start

- You need a company laptop. NorthLink does not run on personal devices.
- Multi-factor authentication must already be set up. See the MFA enrolment guide.
- Your laptop needs to be on a working internet connection.

## Install

NorthLink is installed on all laptops issued since 2024. If you cannot find it, open Company Apps, search for "NorthLink" and choose Install. The install takes about three minutes and may ask you to restart.

## First connection

1. Open NorthLink from the Start menu or Applications folder.
2. Enter the gateway address: vpn.northgate.example
3. Sign in with your Northgate username and password.
4. Approve the request in the NorthKey Authenticator app.
5. Wait until the icon turns green. That means you are connected.

NorthLink remembers the gateway, so you only type it once. It will ask for an MFA approval every 12 hours.

## Which gateway to use

The single address above routes you to the nearest location automatically. Please do not save the addresses of individual gateways, because they can change without notice.

## Good habits

Disconnect when you are finished with internal systems, particularly on video calls, because all traffic goes through the tunnel and calls may suffer. Keep the client updated: Company Apps will prompt you when a new version is available, and versions older than 6.0 are no longer supported.

If the connection fails or keeps dropping, see the VPN Troubleshooting guide, which lists error codes and fixes.
