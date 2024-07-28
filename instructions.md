## Instructions for Building and Working on the Github pages (the project's site):

Prerequisites
1. Ruby and Jekyll: Make sure you have Ruby and Jekyll installed on your system. You can follow the [installation instructions](https://jekyllrb.com/docs/installation/) from the Jekyll documentation.

 Setting Up the Project Locally
1. Clone the Repository
    
2. Install Dependencies:
    ```
    bundle install
    ```

3. Serve the Site Locally:
    ```
    bundle exec jekyll serve
    ```
    - This will start a local server.

Adding New Pages and Tabs
1. Create a New Markdown File:
    - Add a new `.md` file in the appropriate directory (e.g., `Theory` or `notebooks`).
    - Example for creating a new theory page:
        ---
        layout: page
        title: New Theory Page
        permalink: /Theory/New_Theory_Page/
        menubar: theory_menu
        ---
        # New Theory Page
        Content of the new theory page.
        ```

2. Update the Menubar:
    - Open `_data/navigation.yml`.
    - Add an entry for the new page under the appropriate section.


3. Push Changes to GitHub:
    - GitHub Pages will automatically build and deploy the site.

## Instructions for Working on the code:

Supervised Methods
1. STG Model: The original repository for the STG model can be found here: https://github.com/runopti/stg 
2. LSpin Model: The original repository for the LSpin model can be found here: https://github.com/jcyang34/lspin

Unsupervised Methods
1. LSCAE Model: This model includes LSCAE, CAE, and LS model types. The original repository can be found here: https://github.com/jsvir/lscae
2. Gated Laplacian Model: The original repository for the Gated Laplacian model (in TensorFlow) can be found here: https://github.com/Ofirlin/DUFS

Demos Section
- In the `notebooks` folder, you can find three notebooks:
  1. Classification examples of known supervised approaches and the STG model.
  2. Examples of the unsupervised approaches mentioned above.
  3. notebook with a demonstration of LSPIN on a simple regression problem.
