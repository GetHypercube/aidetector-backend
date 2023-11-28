$scriptBlock = {
    param($i)
    curl -X POST -F "file=@test.jpg" http://aidetector-prod-394527615.us-east-1.elb.amazonaws.com/detect -k
}

1..1 | ForEach-Object {
    Start-Job -ScriptBlock $scriptBlock -ArgumentList $_
} | Receive-Job -Wait -AutoRemoveJob