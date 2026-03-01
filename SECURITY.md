# Security Policy

## Reporting a Vulnerability
If you discover a security issue, please do not open a public issue.

Report it privately by contacting the repository owner through GitHub:
- Open a private security advisory in this repository, or
- Contact the maintainer directly through their GitHub profile.

Please include:
- A clear description of the issue
- Steps to reproduce
- Impact assessment
- Suggested mitigation (if known)

We will acknowledge receipt as soon as possible and work on a fix.

## Supported Versions
Security fixes are applied to the latest `main` branch and recent tagged releases used in deployment.

## Deployment Security Notes
- Run this service behind an authenticated reverse proxy when exposed beyond a trusted network.
- Keep `.env` out of version control.
- Use least-privilege file permissions on `${WORKBENCH_DATA_LOCATION}`.
- Regularly update to current image tags and dependencies.
