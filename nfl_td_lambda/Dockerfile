# Use the official AWS Lambda Python 3.9 base image
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies required for building lightgbm
# This is done during the build process, so the final image is clean
RUN yum install -y gcc-c++ cmake

# Copy the requirements file into the image
COPY requirements.txt ./

# Install the python dependencies from requirements.txt
# This will compile lightgbm using the tools we just installed
RUN pip install -r requirements.txt

# Copy the rest of your application code
COPY app.py ./
COPY data_collection.py ./

# Set the command to your handler function
# This tells Lambda what function to run when it's invoked
CMD [ "app.lambda_handler" ]
