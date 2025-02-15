import os

def write_to_file(filename, content):
    """
    Writes the provided content to the specified file.
    If the file doesn't exist, it will be created.
    """
    try:
        with open(filename, 'w') as file:
            file.write(content)
        print(f"Content successfully written to {filename}.")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

def read_from_file(filename):
    """
    Reads the content of the specified file and prints it.
    """
    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist.")
        return
    
    try:
        with open(filename, 'r') as file:
            content = file.read()
        print(f"Content of {filename}:")
        print(content)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

def append_to_file(filename, content):
    """
    Appends the provided content to the specified file.
    """
    try:
        with open(filename, 'a') as file:
            file.write(content)
        print(f"Content successfully appended to {filename}.")
    except Exception as e:
        print(f"An error occurred while appending to the file: {e}")

def delete_file(filename):
    """
    Deletes the specified file if it exists.
    """
    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist.")
        return

    try:
        os.remove(filename)
        print(f"File '{filename}' successfully deleted.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")

if __name__ == "__main__":
    filename = "demo_file.txt"
    
    # Demonstrate writing to a file
    write_to_file(filename, "Hello, this is a demonstration of file handling in Python.\n")

    # Demonstrate reading from a file
    read_from_file(filename)

    # Demonstrate appending to a file
    append_to_file(filename, "Appending a new line to the file.\n")