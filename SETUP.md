Run the pre-setup
`./pre_setup.sh`

From outside this folder(might have to move apptainer.def outide)
`apptainer build --fakeroot Trinity.sif apptainer.def`

Open apptainer shell
`apptainer shell --nv Trinity.sif`

Run the setup to activate env inside the shell
`source /opt/Trinity/activate.sh`

Using the apptainer
- Run the activate script first
- Have a clone of Trinity open. Trinity tries to output stuff to `trinity_output` but it also calls executables that it builds inside its own folder. I changed the code to point at exectuables that will live inside the apptainer, but you need write permissions outside.
- You can add `--writable-tmpfs` to apptainer to allow some writing internally