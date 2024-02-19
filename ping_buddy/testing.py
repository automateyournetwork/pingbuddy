import subprocess
import jc

# Example IP address
ip_address = "8.8.8.8"

# Execute the ping command
command = f'ping -c 4 {ip_address}'
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

# Parse the command output using jc
parsed_output = jc.parse('ping', result.stdout)

print(parsed_output)