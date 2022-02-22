# CS231A Problem Set Template

## Creating a Repository from this Template
### Preferred Method
- Use the option to create repo from template on github website
- Clone the repo: `git clone https://github.com/bcollico/cs231a_ps#.git`
- Rename the folder: `mv cs231a_ps# ps#`
### Alternative Method
- Clone this template: `git clone https://github.com/bcollico/cs231a_ps_template.git`
- Rename the folder: `mv ./cs231a_ps_template ./ps#` where the # corresponds to the problem set
- Navigate to the cloned template: `cd ./ps#`
- Delete the .git folder within the cloned folder: `rm -rf ./.git`
- Initialize a new git repository: `git init .`
- Add the files: `git add .`
- Commit the files: `git commit -m "First commit.`
- Alter the branch name: `git branch -M main`
- Set the upstream: `git remote add origin https://github.com/bcollico/cs231a_ps#.git`
- Push the commit: `git push -u origin main`
## Retrieving Files
- Execute the setup script: `./setup.sh`
- If necessary, first run: `chmod +x setup.sh`
