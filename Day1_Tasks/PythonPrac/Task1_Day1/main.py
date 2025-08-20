from Maths.add import addition 
from Maths.sub import subtraction 
from Maths.mul import multiplication 
from Maths.div import division 

x = int(input("Enter first number: "))
y = int(input("Enter second number: "))

print("addition: ",addition(x,y))
print("subtraction: ",subtraction(x,y))
print("multiplication: ",multiplication(x,y))
print("division: ",division(x,y))