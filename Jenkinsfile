
pipeline {

    
    agent any

    stages {

        stage("Download-data-build-test-fuzzylearn"){

             steps {

                                            sh '''
                                                 docker version
                                                 docker info
                                                 docker build -f Dockerfile.test -t build-image-test-fuzzylearn .
                                            '''


             }

        }

        stage("build-container-test-fuzzylearn") {
            

                 
             steps {

                                            

                                                sh '''
                                                 docker run build-image-test-fuzzylearn
                                                '''
                                            
            
                 }
            }

         
        
        stage("build-image-pypi-fuzzylearn") {
                 
             steps {

                                                 sh '''
                                                 docker version
                                                 docker info
                                                 docker build -f Dockerfile.publish -t build-image-pypi-fuzzylearn .
                                                 '''

            
                 }
            }
    
        stage("build-container-pypi-fuzzylearn") {
            

                 
             steps {

                 withCredentials([
                              usernamePassword(credentialsId: 'twine-login-info-fuzzylearn',
                              usernameVariable: 'username',
                              passwordVariable: 'password',
                              ),
                              usernamePassword(credentialsId: 'fuzzylearn-git-login-with-fine-grained-token',
                              usernameVariable: 'gitusername',
                              passwordVariable: 'gitpassword',
                              )
                              
                                              ]) 

                                              {

                                                 sh '''
                                                 docker run --env username=${username} --env password=${password} --env gitusername=${gitusername}  --env gitpassword=${gitpassword} build-image-pypi-fuzzylearn
                                                 '''
                                              }
            
                 }
            }

    
}

}

