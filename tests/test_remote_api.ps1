$scriptBlock = {
    param($i)
    curl -X POST -F "file=@test_real_image.jpg" https://aidetector-prod.gethypercube.com/detect -k
}

1..1 | ForEach-Object {
    Start-Job -ScriptBlock $scriptBlock -ArgumentList $_
} | Receive-Job -Wait -AutoRemoveJob