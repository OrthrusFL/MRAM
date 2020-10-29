# GithubApp
GithubApp helps you locate buggy code corresponding to bug reports.
Once there is a new bug report raised by users in the issue board, OrthrusFL could give you the suspicious code at method level. 

## Usage
You need to set up a githubApp server by yourself to user OrthrusFL. The process of installation is [here](https://github.com/RAddRiceee/BugPecker/tree/master/GithubApp).
### Initialization
After the installation,you should authorize OrthrusFL to access the repository. The past commits and bug reports will be used to train the model of OrthrusFL. Depends on the size of your repository, the initialization may take several minutes.

As figure 1 shows, statistics of your repository are available after the initialization.
![avatar](https://raw.githubusercontent.com/Tekfei/test/master/init.png)
### Bug Localization
You need to assign your issue a bug label and the corresponding commitId for a new bug report by default is the latest commitId in master branch. If you want to assign a sepcific commitId, you should write in the title in the form of "&commitId:……".

OrthrusFL will be triggered automatically and comment the bug report with suspicious  code.
![avatar](https://raw.githubusercontent.com/Tekfei/test/master/result.png)


If you want to use the githubApp service, you need to set up a [githubApp](https://developer.github.com/) server by yourself.

## Requirements

- Java 1.8
- Tomcat

## Configure

This server calls the Revision Analyzer to initialize your repository. You need to edit the initUrl and UpdateUrl to use the Revision Analyzer in file [HandleIssueServiceImpl](./src/main/java/com/githubApp/service/impl/HandleIssueServiceImpl.java). These parameters refer to the location your analyzer service layed out.

Before use the Matcher and Learner component, you need to edit the resultUrl in file [HandleIssueServiceImpl](./src/main/java/com/githubApp/service/impl/HandleIssueServiceImpl.java). This url should be the same with the ip of the callback url of your githubApp. 

Also you need to change the IP and Port in [LocateBugServiceImpl](./src/main/java/com/githubApp/service/impl/LocateBugServiceImpl.java). These parameters refer the location where your bug locator service layed out.
