on run argv
	tell application "Microsoft Excel"
		return (evaluate name (item 1 of argv))
	end tell
end run
