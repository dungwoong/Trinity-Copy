Run the pre-setup
`./pre_setup.sh`

From outside this folder(might have to move apptainer.def outide)
`apptainer build --fakeroot Trinity.sif apptainer.def`

Open apptainer shell
`apptainer shell --nv Trinity.sif`

Run the setup to activate env inside the shell
`source /opt/Trinity/activate.sh`