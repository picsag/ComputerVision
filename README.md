# ComputerVision

This is a series of computer vision code samples.

## Running instructions

The required packages are under requirements.txt file

Under Windows OS, if an error like the one below is returned when activating a Python virtual environment from Power Shell: <br />
`
\venv\Scripts\activate.ps1 cannot be loaded.
` <br />
then use in Power Shell the following command: <br />
`
Set-ExecutionPolicy Unrestricted -Scope Process
`

## Histogram of Gradients
The histogram of gradients for windows of 8 by 8 pixles are computed using the fuction <br />
`
histogramOfGradients
`