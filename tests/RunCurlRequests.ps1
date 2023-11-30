$scriptBlock = {
    param($i)
    curl -X POST -F "file=@test.jpg" https://aidetector-prod.gethypercube.com/detect -k
}

1..1 | ForEach-Object {
    Start-Job -ScriptBlock $scriptBlock -ArgumentList $_
} | Receive-Job -Wait -AutoRemoveJob