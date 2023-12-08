$scriptBlock = {
    param($i)
    curl -X POST -F "file=@test.jpg" http://localhost:8080/detect -k
}

1..1 | ForEach-Object {
    Start-Job -ScriptBlock $scriptBlock -ArgumentList $_
} | Receive-Job -Wait -AutoRemoveJob