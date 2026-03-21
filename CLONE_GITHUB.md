<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Git LFS Workflow Guide</title>
</head>

<body>

<h1>Git + LFS Workflow (Clone & Development)</h1>

<hr>

<h2>1. Initial Setup (one-time per machine)</h2>

<pre><code>
git lfs install
</code></pre>

<p>
This enables Git LFS hooks globally on your machine.
</p>

<hr>

<h2>2. Cloning the Repository</h2>

<pre><code>
git clone https://github.com/SiracencoSerghei/ML_Project.git
cd ML_Project
</code></pre>

<p>
Git will automatically clone LFS pointer files, not the large binaries.
</p>

<hr>

<h2>3. Downloading LFS Files</h2>

<pre><code>
git lfs pull
</code></pre>

<p>
This downloads the actual large files (e.g. .joblib models) referenced by Git.
</p>

<hr>

<h2>4. Verifying LFS Files</h2>

<pre><code>
git lfs ls-files
</code></pre>

<p>
You should see tracked large files listed here.
</p>

<hr>

<h2>5. Making Changes</h2>

<p>After modifying code or adding files:</p>

<pre><code>
git add .
git commit -m "your message"
</code></pre>

<hr>

<h2>6. Working with Model Files (.joblib)</h2>

<p>
If you replace or update a model file:
</p>

<pre><code>
git add churn/ml/model/churn_model.joblib
git commit -m "update model"
</code></pre>

<p>
Git LFS automatically handles storage — no special commands needed for commits.
</p>

<hr>

<h2>7. Pushing Changes</h2>

<pre><code>
git push origin main
</code></pre>

<p>
LFS files are uploaded automatically during push.
</p>

<hr>

<h2>8. Pulling Updates</h2>

<pre><code>
git pull
git lfs pull
</code></pre>

<p>
Ensures both code and large files are synced.
</p>

<hr>

<h2>9. Common Pitfalls</h2>

<ul>
  <li>Forgetting <code>git lfs install</code> before cloning</li>
  <li>Not running <code>git lfs pull</code> after clone</li>
  <li>Adding large files before enabling LFS tracking</li>
  <li>Committing large binaries without LFS (causes GitHub push rejection)</li>
</ul>

<hr>

<h2>10. Summary Workflow</h2>

<pre><code>
git clone repo
git lfs install
git lfs pull

# work...
git add .
git commit -m "message"
git push origin main
</code></pre>

<hr>

</body>
</html>