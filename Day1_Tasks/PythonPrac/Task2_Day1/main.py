from dotenv import load_dotenv
import os

# Load variables from test.env
load_dotenv(r"C:\Users\ASM9015\Desktop\python practice\Task2_Day1\test.env")

# Access them using os.getenv
name = os.getenv("NAME")
number = os.getenv("KEY")

print(f"Name: {name}")
print(f"Key: {number}")
