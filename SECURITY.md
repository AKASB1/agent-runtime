# Security

Please report a security problem privately through GitHub: on the repository page, open **Security**, then **Report a vulnerability**. If that button is missing, open an issue that asks for a private contact, without details of the problem. Do not put exploit details in a public issue.

The permission layer is policy inside the process, not isolation by the operating system: `run_command` can do whatever its command can do under the account that runs the service. Run the service under an account and in a directory you are willing to expose to the agent ([docs/harness.md](docs/harness.md)).
