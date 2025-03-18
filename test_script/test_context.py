class MyContext:
    def __init__(self, some_input):
        self.some_input = some_input

    def __enter__(self):
        # Initialize or allocate resources based on some_input
        print(f"Entering context with input: {self.some_input}")
        # You might return a value here for use inside the 'with' block
        return self.some_input

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup code goes here
        print(f"Exiting context with input: {self.some_input}")
        # If an exception occurred, you can handle it here
        return False  # Propagate exceptions if any

class MyClass:
    def func(self, some_input):
        # Return an instance of a context manager that uses the given input
        return MyContext(some_input)

# Usage:
a = MyClass()
with a.func("example input") as context_value:
    print(f"Inside the context: {context_value}")